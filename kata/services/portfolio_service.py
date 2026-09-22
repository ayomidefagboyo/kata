"""
Portfolio service for business logic related to portfolio management.
"""
import logging
from typing import List, Dict, Any, Optional
from decimal import Decimal
from datetime import datetime
from supabase import Client

from kata.models.portfolio import Portfolio, Holding, PortfolioSummary
from kata.config.database import get_service_client

logger = logging.getLogger(__name__)


class PortfolioService:
    """Service class for portfolio operations."""
    
    def __init__(self):
        self.db: Client = get_service_client()
    
    async def sync_portfolio_from_alchemy(
        self, 
        user_id: str, 
        alchemy_data: Dict[str, Any]
    ) -> Portfolio:
        """
        Sync portfolio data from Alchemy API response.
        
        Args:
            user_id: User ID
            alchemy_data: Data from Alchemy API containing tokens and balances
            
        Returns:
            Updated Portfolio object
        """
        try:
            # Get user's portfolio
            portfolio_response = self.db.from_("portfolios").select("*").eq("user_id", user_id).execute()
            
            if not portfolio_response.data:
                raise ValueError(f"Portfolio not found for user {user_id}")
            
            portfolio_id = portfolio_response.data[0]["id"]
            
            # Clear existing holdings (we'll replace with fresh data)
            self.db.from_("holdings").delete().eq("portfolio_id", portfolio_id).execute()
            
            # Process Alchemy token data
            total_value = Decimal("0")
            holdings_created = 0
            
            for token in alchemy_data.get("tokens", []):
                if float(token.get("balance_formatted", 0)) > 0:
                    holding_data = {
                        "portfolio_id": portfolio_id,
                        "token_address": token.get("address", ""),
                        "token_symbol": token.get("symbol", ""),
                        "amount": Decimal(str(token.get("balance_formatted", 0))),
                        "value_usd": Decimal(str(token.get("value_usd", 0))),
                        "pnl_usd": Decimal("0"),  # Calculate based on historical data if available
                        "protocol": "base",  # Default to Base network
                        "position_type": "spot"
                    }
                    
                    self.db.from_("holdings").insert(holding_data).execute()
                    total_value += holding_data["value_usd"]
                    holdings_created += 1
            
            # Update portfolio totals
            portfolio_update = {
                "total_value_usd": float(total_value),
                "last_updated": datetime.now().isoformat()
            }
            
            updated_portfolio_response = self.db.from_("portfolios").update(portfolio_update).eq("id", portfolio_id).execute()
            
            if updated_portfolio_response.data:
                portfolio = Portfolio(**updated_portfolio_response.data[0])
                logger.info(f"Synced portfolio for user {user_id}: {holdings_created} holdings, ${total_value} total value")
                return portfolio
            else:
                raise ValueError("Failed to update portfolio")
                
        except Exception as e:
            logger.error(f"Error syncing portfolio from Alchemy for user {user_id}: {e}")
            raise
    
    async def calculate_portfolio_metrics(self, user_id: str) -> Dict[str, Any]:
        """
        Calculate portfolio performance metrics.
        
        Args:
            user_id: User ID
            
        Returns:
            Dictionary containing calculated metrics
        """
        try:
            # Get historical performance data
            metrics_response = self.db.from_("performance_metrics").select("*").eq("user_id", user_id).order("metric_date", desc=True).limit(30).execute()
            
            if not metrics_response.data:
                return self._default_metrics()
            
            metrics_data = metrics_response.data
            current_value = Decimal(str(metrics_data[0]["portfolio_value_usd"]))
            
            # Calculate various time period returns
            returns = self._calculate_returns(metrics_data, current_value)
            
            # Calculate volatility and risk metrics
            daily_returns = self._calculate_daily_returns(metrics_data)
            volatility = self._calculate_volatility(daily_returns)
            sharpe_ratio = self._calculate_sharpe_ratio(daily_returns, volatility)
            max_drawdown = self._calculate_max_drawdown(metrics_data)
            
            metrics = {
                "current_value": float(current_value),
                "returns": returns,
                "volatility": float(volatility),
                "sharpe_ratio": float(sharpe_ratio) if sharpe_ratio else None,
                "max_drawdown": float(max_drawdown),
                "last_updated": datetime.now().isoformat()
            }
            
            logger.info(f"Calculated portfolio metrics for user {user_id}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics for user {user_id}: {e}")
            return self._default_metrics()
    
    async def update_daily_performance_metric(self, user_id: str) -> bool:
        """
        Update daily performance metric for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get current portfolio value
            portfolio_response = self.db.from_("portfolios").select("*").eq("user_id", user_id).execute()
            
            if not portfolio_response.data:
                logger.warning(f"Portfolio not found for user {user_id}")
                return False
            
            current_value = Decimal(str(portfolio_response.data[0]["total_value_usd"]))
            today = datetime.now().date()
            
            # Get yesterday's value for daily P&L calculation
            yesterday_response = self.db.from_("performance_metrics").select("portfolio_value_usd").eq("user_id", user_id).eq("metric_date", (today - timedelta(days=1)).isoformat()).execute()
            
            yesterday_value = Decimal("0")
            if yesterday_response.data:
                yesterday_value = Decimal(str(yesterday_response.data[0]["portfolio_value_usd"]))
            
            daily_pnl = current_value - yesterday_value
            
            # Get today's trade count
            trades_response = self.db.from_("trades").select("id", count="exact").eq("user_id", user_id).gte("created_at", today.isoformat()).execute()
            trades_count = trades_response.count if hasattr(trades_response, 'count') else 0
            
            # Calculate win rate (simplified - would need actual P&L per trade)
            win_rate = Decimal("0.0")  # No real data available yet
            
            # Upsert performance metric
            metric_data = {
                "user_id": user_id,
                "agent_type": "overall",
                "metric_date": today.isoformat(),
                "portfolio_value_usd": float(current_value),
                "daily_pnl_usd": float(daily_pnl),
                "trades_count": trades_count,
                "win_rate": float(win_rate)
            }
            
            # Try to update existing record first
            existing_response = self.db.from_("performance_metrics").select("id").eq("user_id", user_id).eq("metric_date", today.isoformat()).eq("agent_type", "overall").execute()
            
            if existing_response.data:
                # Update existing
                self.db.from_("performance_metrics").update(metric_data).eq("id", existing_response.data[0]["id"]).execute()
            else:
                # Insert new
                self.db.from_("performance_metrics").insert(metric_data).execute()
            
            logger.info(f"Updated daily performance metric for user {user_id}: ${current_value} (${daily_pnl:+.2f})")
            return True
            
        except Exception as e:
            logger.error(f"Error updating daily performance metric for user {user_id}: {e}")
            return False
    
    def _default_metrics(self) -> Dict[str, Any]:
        """Return default metrics when no data is available."""
        return {
            "current_value": 0.0,
            "returns": {
                "daily": 0.0,
                "weekly": 0.0,
                "monthly": 0.0,
                "yearly": 0.0
            },
            "volatility": 0.0,
            "sharpe_ratio": None,
            "max_drawdown": 0.0,
            "last_updated": datetime.now().isoformat()
        }
    
    def _calculate_returns(self, metrics_data: List[Dict], current_value: Decimal) -> Dict[str, float]:
        """Calculate returns for different time periods."""
        returns = {"daily": 0.0, "weekly": 0.0, "monthly": 0.0, "yearly": 0.0}
        
        if len(metrics_data) > 1:
            # Daily return
            previous_value = Decimal(str(metrics_data[1]["portfolio_value_usd"]))
            if previous_value > 0:
                returns["daily"] = float((current_value - previous_value) / previous_value * 100)
        
        if len(metrics_data) > 7:
            # Weekly return
            week_ago_value = Decimal(str(metrics_data[7]["portfolio_value_usd"]))
            if week_ago_value > 0:
                returns["weekly"] = float((current_value - week_ago_value) / week_ago_value * 100)
        
        if len(metrics_data) > 30:
            # Monthly return
            month_ago_value = Decimal(str(metrics_data[30]["portfolio_value_usd"]))
            if month_ago_value > 0:
                returns["monthly"] = float((current_value - month_ago_value) / month_ago_value * 100)
        
        # Yearly return (if we have enough data)
        if len(metrics_data) >= 365:
            year_ago_value = Decimal(str(metrics_data[365]["portfolio_value_usd"]))
            if year_ago_value > 0:
                returns["yearly"] = float((current_value - year_ago_value) / year_ago_value * 100)
        
        return returns
    
    def _calculate_daily_returns(self, metrics_data: List[Dict]) -> List[float]:
        """Calculate daily returns from metrics data."""
        daily_returns = []
        
        for i in range(len(metrics_data) - 1):
            current = Decimal(str(metrics_data[i]["portfolio_value_usd"]))
            previous = Decimal(str(metrics_data[i + 1]["portfolio_value_usd"]))
            
            if previous > 0:
                daily_return = float((current - previous) / previous)
                daily_returns.append(daily_return)
        
        return daily_returns
    
    def _calculate_volatility(self, daily_returns: List[float]) -> Decimal:
        """Calculate portfolio volatility (standard deviation of returns)."""
        if len(daily_returns) < 2:
            return Decimal("0")
        
        mean_return = sum(daily_returns) / len(daily_returns)
        variance = sum((r - mean_return) ** 2 for r in daily_returns) / (len(daily_returns) - 1)
        volatility = variance ** 0.5
        
        return Decimal(str(volatility))
    
    def _calculate_sharpe_ratio(self, daily_returns: List[float], volatility: Decimal) -> Optional[Decimal]:
        """Calculate Sharpe ratio (risk-adjusted return)."""
        if len(daily_returns) < 2 or volatility == 0:
            return None
        
        mean_return = sum(daily_returns) / len(daily_returns)
        risk_free_rate = 0.02 / 365  # Assume 2% annual risk-free rate
        
        excess_return = mean_return - risk_free_rate
        sharpe_ratio = excess_return / float(volatility)
        
        return Decimal(str(sharpe_ratio))
    
    def _calculate_max_drawdown(self, metrics_data: List[Dict]) -> Decimal:
        """Calculate maximum drawdown."""
        if len(metrics_data) < 2:
            return Decimal("0")
        
        values = [Decimal(str(metric["portfolio_value_usd"])) for metric in metrics_data]
        values.reverse()  # Get chronological order
        
        peak = values[0]
        max_drawdown = Decimal("0")
        
        for value in values[1:]:
            if value > peak:
                peak = value
            else:
                drawdown = (peak - value) / peak
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
        
        return max_drawdown * 100  # Return as percentage


# Global service instance
portfolio_service = PortfolioService()