"""
Privy-Hyperliquid Integration Service

This service handles the integration between Privy wallet management and Hyperliquid trading.
It manages wallet derivation, private key handling, and order signing for live trading.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
import hashlib
import hmac
import base64
from eth_account import Account
from eth_account.messages import encode_defunct
from web3 import Web3

logger = logging.getLogger(__name__)


@dataclass
class PrivyUserInfo:
    """Privy user information."""
    user_id: str
    wallet_address: str
    private_key: Optional[str] = None
    is_verified: bool = False


class PrivyHyperliquidService:
    """
    Service for integrating Privy wallet management with Hyperliquid trading.
    
    This service handles:
    - Privy wallet authentication and verification
    - Private key derivation for Hyperliquid trading
    - Order signing and authentication
    - Wallet delegation and management
    """
    
    def __init__(self, privy_app_id: str, privy_app_secret: str):
        """Initialize the Privy-Hyperliquid service."""
        self.privy_app_id = privy_app_id
        self.privy_app_secret = privy_app_secret
        self.web3 = Web3()
        
        # Store authenticated users
        self.authenticated_users: Dict[str, PrivyUserInfo] = {}

        logger.info("PrivyHyperliquidService initialized")

    async def authenticate_user_with_privy(self, access_token: str) -> Dict[str, Any]:
        """
        Authenticate user with Privy and setup Hyperliquid trading access.

        Args:
            access_token: Privy access token from frontend

        Returns:
            Authentication result with user info and trading readiness
        """
        try:
            # Step 1: Verify Privy access token
            user_info = await self._verify_privy_token(access_token)
            if not user_info:
                return {
                    "success": False,
                    "error": "Invalid Privy access token"
                }
            
            # Step 2: Extract wallet information
            wallet_address = user_info.get("wallet_address")
            if not wallet_address:
                return {
                    "success": False,
                    "error": "No wallet address found in Privy user info"
                }
            
            # Step 3: Derive private key for Hyperliquid trading
            private_key = await self._derive_private_key(user_info)
            if not private_key:
                return {
                    "success": False,
                    "error": "Failed to derive private key for trading"
                }
            
            # Step 4: Store user information
            privy_user = PrivyUserInfo(
                user_id=user_info["user_id"],
                wallet_address=wallet_address,
                private_key=private_key,
                is_verified=True
            )
            
            self.authenticated_users[user_info["user_id"]] = privy_user
            
            # Step 5: Check Hyperliquid account status
            hyperliquid_status = await self._check_hyperliquid_account(wallet_address)

            return {
                "success": True,
                "user_id": user_info["user_id"],
                "wallet_address": wallet_address,
                "hyperliquid_account_active": hyperliquid_status["account_exists"],
                "balance_usdc": hyperliquid_status.get("balance", 0),
                "ready_for_trading": hyperliquid_status["account_exists"] and hyperliquid_status.get("balance", 0) > 0,
                "funding_needed": not hyperliquid_status["account_exists"] or hyperliquid_status.get("balance", 0) == 0,
                "message": "Successfully authenticated with Privy and Hyperliquid"
            }

        except Exception as e:
            logger.error(f"Error authenticating user with Privy: {e}")
            return {
                "success": False,
                "error": f"Authentication failed: {str(e)}"
            }

    async def _verify_privy_token(self, access_token: str) -> Optional[Dict[str, Any]]:
        """
        Verify Privy access token and extract user information.
        
        Args:
            access_token: Privy access token

        Returns:
            User information if valid, None otherwise
        """
        try:
            logger.warning("Legacy Privy token verification is disabled; use PrivyAuthService for real verification")
            return None

        except Exception as e:
            logger.error(f"Error verifying Privy token: {e}")
            return None

    async def _derive_private_key(self, user_info: Dict[str, Any]) -> Optional[str]:
        """
        Derive private key for Hyperliquid trading from Privy user info.
        
        In a real implementation, this would:
        1. Use Privy's wallet management to get the private key
        2. Or derive it from the user's wallet using secure methods

        Args:
            user_info: User information from Privy

        Returns:
            Private key string if successful, None otherwise
        """
        try:
            logger.warning("Legacy private key derivation is disabled; use real Privy wallet signing/delegation")
            return None
            
        except Exception as e:
            logger.error(f"Error deriving private key: {e}")
            return None
    
    async def _check_hyperliquid_account(self, wallet_address: str) -> Dict[str, Any]:
        """
        Check if the wallet address has an active Hyperliquid account.
        
        Args:
            wallet_address: Wallet address to check
            
        Returns:
            Account status information
        """
        try:
            logger.warning("Legacy Hyperliquid account simulation is disabled for %s", wallet_address)
            return {
                "account_exists": False,
                "balance": 0.0,
                "margin_used": 0.0,
                "unrealized_pnl": 0.0,
                "positions_count": 0,
                "error": "Legacy account check disabled. Use the live Hyperliquid service for real account state."
            }

        except Exception as e:
            logger.error(f"Error checking Hyperliquid account: {e}")
            return {
                "account_exists": False,
                "balance": 0.0,
                "margin_used": 0.0,
                "unrealized_pnl": 0.0,
                "positions_count": 0,
                "error": str(e)
            }
    
    def get_user_private_key(self, user_id: str) -> Optional[str]:
        """
        Get the private key for a user's wallet.

        Args:
            user_id: User ID

        Returns:
            Private key if user is authenticated, None otherwise
        """
        user_info = self.authenticated_users.get(user_id)
        if user_info and user_info.is_verified:
            return user_info.private_key
        return None
    
    def get_user_wallet_address(self, user_id: str) -> Optional[str]:
        """
        Get the wallet address for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Wallet address if user is authenticated, None otherwise
        """
        user_info = self.authenticated_users.get(user_id)
        if user_info and user_info.is_verified:
            return user_info.wallet_address
        return None
    
    def is_user_authenticated(self, user_id: str) -> bool:
        """
        Check if a user is authenticated and ready for trading.
        
        Args:
            user_id: User ID
            
        Returns:
            True if user is authenticated, False otherwise
        """
        user_info = self.authenticated_users.get(user_id)
        return user_info is not None and user_info.is_verified
    
    async def sign_hyperliquid_order(self, user_id: str, order_data: Dict[str, Any]) -> Optional[str]:
        """
        Sign a Hyperliquid order with the user's private key.
        
        Args:
            user_id: User ID
            order_data: Order data to sign
            
        Returns:
            Signed order data if successful, None otherwise
        """
        try:
            private_key = self.get_user_private_key(user_id)
            if not private_key:
                logger.error(f"No private key found for user {user_id}")
                return None
            
            # Create account from private key
            account = Account.from_key(private_key)
            
            # In a real implementation, this would use Hyperliquid's signing protocol
            # For now, we'll create a basic signature
            
            # Serialize order data
            order_json = json.dumps(order_data, sort_keys=True)
            
            # Create message hash
            message_hash = hashlib.sha256(order_json.encode()).hexdigest()
            
            # Sign the message
            message = encode_defunct(hexstr=message_hash)
            signed_message = account.sign_message(message)
            
            # Return the signature
            return signed_message.signature.hex()

        except Exception as e:
            logger.error(f"Error signing Hyperliquid order: {e}")
            return None
    
    async def allocate_funds_to_agent(
        self,
        user_id: str,
        agent_type: str,
        amount: float,
        eth_amount: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Allocate funds to a trading agent for automated trading.

        Funding must be performed through real on-chain transactions before an
        allocation is created. This legacy helper intentionally fails closed
        until a real transaction executor is wired in.

        Args:
            user_id: User identifier
            agent_type: Type of agent (yuki, sakura, ryu)
            amount: USDC amount to allocate after bridging
            eth_amount: ETH amount to bridge (optional)

        Returns:
            Allocation result with success status and details
        """
        logger.warning(
            "Blocked legacy Privy-Hyperliquid allocation for %s: real on-chain funding executor is not implemented",
            user_id,
        )
        return {
            "success": False,
            "error": (
                "Automated funding is disabled until real on-chain swap, bridge, and Hyperliquid deposit "
                "transactions are implemented. Fund Hyperliquid with real USDC, then allocate to Yuki."
            )
        }

    async def get_funding_instructions(self, user_id: str) -> Dict[str, Any]:
        """
        Get funding instructions for the user's Hyperliquid account.
        
        Args:
            user_id: User ID
            
        Returns:
            Funding instructions and account details
        """
        try:
            wallet_address = self.get_user_wallet_address(user_id)
            if not wallet_address:
                return {
                    "success": False,
                    "error": "User not authenticated"
                }
            
            # In a real implementation, this would provide actual funding instructions
            # for the user's Hyperliquid account

            return {
                "success": True,
                "wallet_address": wallet_address,
                "funding_methods": [
                    {
                        "method": "USDC Bridge",
                        "description": "Bridge USDC from Base to Arbitrum for Hyperliquid",
                        "min_amount": 50.0,
                        "estimated_time": "1-2 minutes"
                    },
                    {
                        "method": "Direct Deposit",
                        "description": "Direct USDC deposit to Hyperliquid",
                        "min_amount": 10.0,
                        "estimated_time": "Instant"
                    }
                ],
                "current_balance": 0.0,
                "minimum_balance": 50.0
            }

        except Exception as e:
            logger.error(f"Error getting funding instructions: {e}")
            return {
                "success": False,
                "error": f"Failed to get funding instructions: {str(e)}"
            }
    
    def cleanup_user_session(self, user_id: str) -> bool:
        """
        Clean up user session and remove sensitive data.
        
        Args:
            user_id: User ID
            
        Returns:
            True if cleanup successful, False otherwise
        """
        try:
            if user_id in self.authenticated_users:
                # Clear private key from memory
                user_info = self.authenticated_users[user_id]
                user_info.private_key = None
                user_info.is_verified = False
                
                # Remove from authenticated users
                del self.authenticated_users[user_id]
                
                logger.info(f"Cleaned up session for user {user_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error cleaning up user session: {e}")
            return False

    async def _bridge_eth_to_usdc(
        self,
        user_id: str,
        eth_amount: float,
        expected_usdc: float,
        wallet_address: str
    ) -> Dict[str, Any]:
        """
        Bridge ETH to USDC using LiFi integration.

        Args:
            user_id: User identifier
            eth_amount: Amount of ETH to bridge
            expected_usdc: Expected USDC amount after bridge
            wallet_address: User's wallet address

        Returns:
            Bridge result with transaction details
        """
        try:
            logger.info(f"Initiating ETH to USDC bridge for user {user_id}: {eth_amount} ETH -> ~{expected_usdc} USDC")

            # Get LiFi bridge service
            from kata.services.lifi_bridge_service import get_lifi_service
            lifi_service = get_lifi_service()

            # Get bridge quote
            quote = await lifi_service.get_bridge_quote(
                from_chain="base",
                to_chain="arbitrum",
                from_token="ETH",
                to_token="USDC",
                amount=eth_amount,
                wallet_address=wallet_address
            )

            if not quote:
                return {
                    "success": False,
                    "error": "Failed to get bridge quote from LiFi"
                }

            return {
                "success": False,
                "error": (
                    "Real ETH to USDC bridge execution is not implemented yet. "
                    "Quote generated only; no on-chain transaction was submitted."
                ),
                "estimated_usdc": quote.output_amount,
                "bridge_provider": quote.tool_name,
                "estimated_time": quote.estimated_time,
                "total_fees": quote.total_fees
            }

        except Exception as e:
            logger.error(f"Error bridging ETH to USDC: {e}")
            return {
                "success": False,
                "error": f"Bridge failed: {str(e)}"
            }

    async def _setup_hyperliquid_account(self, user_id: str, wallet_address: str) -> Dict[str, Any]:
        """
        Setup or verify Hyperliquid account for the user.

        Args:
            user_id: User identifier
            wallet_address: User's wallet address

        Returns:
            Setup result with account details
        """
        try:
            logger.info(f"Setting up Hyperliquid account for user {user_id} at {wallet_address}")

            # Check if account already exists
            account_status = await self._check_hyperliquid_account(wallet_address)

            if account_status.get("account_exists"):
                logger.info(f"Hyperliquid account already exists for {wallet_address}")
                return {
                    "success": True,
                    "account_address": wallet_address,
                    "existing_account": True,
                    "balance": account_status.get("balance", 0)
                }

            return {
                "success": False,
                "account_address": wallet_address,
                "existing_account": False,
                "balance": 0.0,
                "error": "No Hyperliquid account state found. Deposit real USDC to activate the account."
            }

        except Exception as e:
            logger.error(f"Error setting up Hyperliquid account: {e}")
            return {
                "success": False,
                "error": f"Account setup failed: {str(e)}"
            }

    async def _transfer_to_hyperliquid(
        self,
        user_id: str,
        amount: float,
        wallet_address: str
    ) -> Dict[str, Any]:
        """
        Transfer USDC to user's Hyperliquid account.

        Args:
            user_id: User identifier
            amount: USDC amount to transfer
            wallet_address: User's wallet address

        Returns:
            Transfer result with transaction details
        """
        try:
            logger.info(f"Transferring {amount} USDC to Hyperliquid for user {user_id}")

            # Get user's private key for signing
            private_key = self.get_user_private_key(user_id)
            if not private_key:
                return {
                    "success": False,
                    "error": "No private key available for user"
                }

            return {
                "success": False,
                "error": (
                    "Real USDC transfer to Hyperliquid is not implemented in this service. "
                    "No transaction was submitted."
                ),
                "account_address": wallet_address,
                "amount_requested": amount
            }

        except Exception as e:
            logger.error(f"Error transferring to Hyperliquid: {e}")
            return {
                "success": False,
                "error": f"Transfer failed: {str(e)}"
            }

    async def _initialize_agent_trading(
        self,
        user_id: str,
        agent_type: str,
        allocated_amount: float,
        wallet_address: str
    ) -> Dict[str, Any]:
        """
        Initialize agent trading with allocated funds.

        Args:
            user_id: User identifier
            agent_type: Type of agent (yuki, sakura, ryu)
            allocated_amount: Amount allocated to agent
            wallet_address: User's wallet address

        Returns:
            Agent initialization result
        """
        try:
            logger.info(f"Initializing {agent_type} agent for user {user_id} with {allocated_amount} USDC")

            return {
                "success": False,
                "error": "Legacy agent initialization is disabled until real funding is verified.",
                "agent_type": agent_type,
                "allocated_amount": allocated_amount,
                "trading_active": False,
            }

        except Exception as e:
            logger.error(f"Error initializing agent trading: {e}")
            return {
                "success": False,
                "error": f"Agent initialization failed: {str(e)}"
            }


# Factory function for creating service instance
def get_privy_hyperliquid_service() -> PrivyHyperliquidService:
    """Get or create PrivyHyperliquidService instance."""
    from kata.config.settings import settings

    return PrivyHyperliquidService(
        privy_app_id=settings.PRIVY_APP_ID,
        privy_app_secret=settings.PRIVY_APP_SECRET
    )
