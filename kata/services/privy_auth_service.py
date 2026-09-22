"""
Privy Authentication Service for Hyperliquid Integration

This service handles Privy wallet authentication using the official Privy SDK
and creates the necessary wallet connections for Hyperliquid trading.
"""

import asyncio
import logging
import json
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import httpx
from dataclasses import dataclass

from privy import PrivyAPI
from privy.types import User, Wallet

from hyperliquid.exchange import Exchange
from hyperliquid.utils.signing import sign_l1_action

logger = logging.getLogger(__name__)


@dataclass
class PrivyWalletInfo:
    """Privy wallet information."""
    user_id: str
    wallet_address: str
    wallet_type: str
    verified: bool
    created_at: datetime
    access_token: str


@dataclass 
class HyperliquidWalletConnection:
    """Hyperliquid wallet connection details."""
    wallet_address: str
    exchange_client: Exchange
    info_client: Any
    testnet: bool
    connected_at: datetime


class PrivyAuthService:
    """
    Service for handling Privy authentication and delegated wallet actions.

    This service manages:
    - Privy token verification and user authentication
    - Delegated wallet actions for AI agents
    - Session signer management for autonomous trading
    - Wallet permission validation and security
    """

    def __init__(self, privy_app_id: str, privy_app_secret: str, testnet: bool = False):
        """Initialize Privy authentication service."""
        self.privy_app_id = privy_app_id
        self.privy_app_secret = privy_app_secret
        self.testnet = testnet

        # Initialize official Privy client
        self.privy_client = PrivyAPI(
            app_id=privy_app_id,
            app_secret=privy_app_secret
        )

        # Base URLs
        self.hyperliquid_base_url = "https://api.hyperliquid-testnet.xyz" if testnet else "https://api.hyperliquid.xyz"

        # Active connections and delegated wallets
        self.active_connections: Dict[str, HyperliquidWalletConnection] = {}
        self.wallet_cache: Dict[str, PrivyWalletInfo] = {}
        self.delegated_wallets: Dict[str, Dict[str, Any]] = {}  # user_id -> delegated wallet info

        logger.info(f"PrivyAuthService initialized with official SDK (testnet: {testnet})")
    
    async def verify_privy_token(self, access_token: str) -> PrivyWalletInfo:
        """
        Verify Privy access token and extract wallet information using official SDK.

        Args:
            access_token: Privy access token from frontend

        Returns:
            PrivyWalletInfo with verified wallet details
        """
        try:
            # Check cache first
            if access_token in self.wallet_cache:
                cached_info = self.wallet_cache[access_token]
                # Check if cache is still valid (1 hour)
                if (datetime.now() - cached_info.created_at).seconds < 3600:
                    logger.debug("Returning cached Privy wallet info")
                    return cached_info

            # Verify the token signature, issuer, audience, and expiration using
            # Privy's SDK before trusting the user ID in its claims.
            try:
                claims = await asyncio.to_thread(
                    self.privy_client.users.verify_access_token,
                    auth_token=access_token,
                )
                # privy-client 0.6 exposes AccessTokenClaims as a TypedDict,
                # while older releases returned an object. Accept both shapes
                # without ever falling back to decoding an unverified token.
                user_id = (
                    claims.get("user_id")
                    if isinstance(claims, dict)
                    else getattr(claims, "user_id", None)
                )
                if not user_id:
                    raise ValueError("Verified Privy token did not include a user ID")

                # Verify user exists using the Privy API
                user_data = await asyncio.to_thread(
                    self.privy_client.users.get,
                    user_id
                )

                # Extract wallet information from user data
                primary_wallet = None
                ethereum_wallets = []
                for linked_account in user_data.linked_accounts or []:
                    if (hasattr(linked_account, 'type') and
                        linked_account.type == 'wallet' and
                        hasattr(linked_account, 'chain_type') and
                        linked_account.chain_type == 'ethereum'):
                        ethereum_wallets.append(linked_account)

                # Base deposits and agent funding use the embedded Kata
                # wallet. An external login wallet can appear first in Privy's
                # linked-account list, and Ryu adds a second chain-specific
                # wallet, so account ordering is not a safe selector.
                primary_wallet = next(
                    (
                        wallet
                        for wallet in ethereum_wallets
                        if getattr(wallet, 'imported', None) is not True
                        and getattr(wallet, 'wallet_client_type', 'privy')
                        in {'privy', 'privy-v2'}
                    ),
                    None,
                )
                if primary_wallet is None and ethereum_wallets:
                    primary_wallet = ethereum_wallets[0]

                if not primary_wallet:
                    raise ValueError("No Ethereum wallet found in Privy account")

                wallet_info = PrivyWalletInfo(
                    user_id=user_data.id,
                    wallet_address=primary_wallet.address,
                    wallet_type=getattr(primary_wallet, 'wallet_type', 'privy'),
                    verified=getattr(primary_wallet, 'verified', False),
                    created_at=datetime.now(),
                    access_token=access_token
                )

                # Cache the result
                self.wallet_cache[access_token] = wallet_info

                logger.info(f"Privy token verified for wallet: {wallet_info.wallet_address}")
                return wallet_info

            except Exception as sdk_error:
                logger.error(f"Privy SDK error: {sdk_error}")
                raise ValueError(f"Token verification failed: {sdk_error}")

        except Exception as e:
            logger.error(f"Error verifying Privy token: {e}")
            raise ValueError(f"Token verification failed: {e}")
    
    async def create_hyperliquid_connection(self, wallet_info: PrivyWalletInfo) -> HyperliquidWalletConnection:
        """
        Create authenticated Hyperliquid connection for trading.
        
        Args:
            wallet_info: Verified Privy wallet information
            
        Returns:
            HyperliquidWalletConnection ready for trading
        """
        try:
            # Check if connection already exists
            if wallet_info.wallet_address in self.active_connections:
                existing_conn = self.active_connections[wallet_info.wallet_address]
                logger.debug(f"Returning existing Hyperliquid connection for {wallet_info.wallet_address}")
                return existing_conn
            
            # Create Hyperliquid connection with real wallet integration
            from kata.services.hyperliquid_client_factory import create_info_client

            info_client = create_info_client(base_url=self.hyperliquid_base_url if not self.testnet else None)

            # Create exchange client with Privy wallet integration
            # Note: This will require implementing Privy wallet signing for Hyperliquid
            exchange_client = None  # Will be implemented with Privy wallet signing
            
            connection = HyperliquidWalletConnection(
                wallet_address=wallet_info.wallet_address,
                exchange_client=exchange_client,
                info_client=info_client,
                testnet=self.testnet,
                connected_at=datetime.now()
            )
            
            # Store active connection
            self.active_connections[wallet_info.wallet_address] = connection
            
            logger.info(f"Hyperliquid connection created for wallet: {wallet_info.wallet_address}")
            return connection
            
        except Exception as e:
            logger.error(f"Error creating Hyperliquid connection: {e}")
            raise ValueError(f"Connection creation failed: {e}")
    
    async def authenticate_and_connect(self, access_token: str) -> Dict[str, Any]:
        """
        Complete authentication flow: verify Privy token and create Hyperliquid connection.
        
        Args:
            access_token: Privy access token from frontend
            
        Returns:
            Complete authentication and connection info
        """
        try:
            # Step 1: Verify Privy token
            wallet_info = await self.verify_privy_token(access_token)
            
            # Step 2: Create Hyperliquid connection
            hl_connection = await self.create_hyperliquid_connection(wallet_info)
            
            # Step 3: Test connection
            connection_status = await self._test_hyperliquid_connection(hl_connection)
            
            return {
                "status": "success",
                "user_id": wallet_info.user_id,
                "wallet_address": wallet_info.wallet_address,
                "wallet_verified": wallet_info.verified,
                "hyperliquid_connected": connection_status["connected"],
                "trading_enabled": connection_status["trading_enabled"],
                "testnet": self.testnet,
                "connected_at": hl_connection.connected_at.isoformat(),
                "capabilities": {
                    "market_data": True,
                    "trading": connection_status["trading_enabled"],
                    "portfolio_data": connection_status["connected"]
                }
            }
            
        except Exception as e:
            logger.error(f"Authentication and connection failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _test_hyperliquid_connection(self, connection: HyperliquidWalletConnection) -> Dict[str, Any]:
        """Test Hyperliquid connection and return capabilities."""
        try:
            # Test info client
            meta = connection.info_client.meta()
            market_data_available = len(meta.get('universe', [])) > 0
            
            # Test if we can get user state (requires wallet)
            portfolio_data_available = False
            if connection.wallet_address:
                try:
                    user_state = connection.info_client.user_state(connection.wallet_address)
                    portfolio_data_available = user_state is not None
                except:
                    portfolio_data_available = False
            
            # Trading requires exchange client with wallet
            trading_enabled = connection.exchange_client is not None
            
            return {
                "connected": True,
                "market_data_available": market_data_available,
                "portfolio_data_available": portfolio_data_available,
                "trading_enabled": trading_enabled,
                "markets_count": len(meta.get('universe', [])),
                "test_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Hyperliquid connection test failed: {e}")
            return {
                "connected": False,
                "error": str(e),
                "market_data_available": False,
                "portfolio_data_available": False,
                "trading_enabled": False
            }
    
    async def get_user_portfolio(self, wallet_address: str) -> Dict[str, Any]:
        """Get user portfolio data from Hyperliquid."""
        try:
            if wallet_address not in self.active_connections:
                raise ValueError("No active connection for wallet")
            
            connection = self.active_connections[wallet_address]
            
            # Get user state
            user_state = connection.info_client.user_state(wallet_address)
            
            if not user_state:
                return {
                    "wallet_address": wallet_address,
                    "account_exists": False,
                    "message": "No account found on Hyperliquid"
                }
            
            # Process portfolio data
            cross_margin = user_state.get('crossMarginSummary', {})
            asset_positions = user_state.get('assetPositions', [])
            
            # Extract positions
            positions = []
            for asset_pos in asset_positions:
                position = asset_pos.get('position', {})
                size = float(position.get('szi', 0))
                
                if abs(size) > 1e-8:  # Has position
                    positions.append({
                        "symbol": asset_pos.get('coin'),
                        "size": size,
                        "side": "long" if size > 0 else "short",
                        "entry_price": float(position.get('entryPx', 0)),
                        "unrealized_pnl": float(position.get('unrealizedPnl', 0)),
                        "margin_used": float(position.get('marginUsed', 0))
                    })
            
            return {
                "wallet_address": wallet_address,
                "account_exists": True,
                "account_value": float(cross_margin.get('accountValue', 0)),
                "total_margin_used": float(cross_margin.get('totalMarginUsed', 0)),
                "total_ntl_pos": float(cross_margin.get('totalNtlPos', 0)),
                "withdrawable": float(user_state.get('withdrawable', 0)),
                "positions": positions,
                "positions_count": len(positions),
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error fetching user portfolio: {e}")
            return {
                "wallet_address": wallet_address,
                "error": str(e),
                "account_exists": False
            }
    
    def disconnect_user(self, wallet_address: str) -> bool:
        """Disconnect user and cleanup connections."""
        try:
            if wallet_address in self.active_connections:
                del self.active_connections[wallet_address]
                logger.info(f"Disconnected user: {wallet_address}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error disconnecting user: {e}")
            return False
    
    def get_connection_info(self, wallet_address: str) -> Optional[Dict[str, Any]]:
        """Get connection information for a wallet."""
        if wallet_address not in self.active_connections:
            return None

    async def get_delegated_wallets(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get delegated wallets for a user using Privy API.

        Args:
            user_id: Privy user ID

        Returns:
            List of delegated wallet information
        """
        try:
            headers = {
                'Authorization': f'Bearer {self.privy_app_secret}',
                'privy-app-id': self.privy_app_id,
                'Content-Type': 'application/json'
            }

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.privy_api_url}/v1/users/{user_id}/delegated_wallets",
                    headers=headers
                )

                if response.status_code == 200:
                    delegated_data = response.json()
                    self.delegated_wallets[user_id] = delegated_data
                    logger.info(f"Retrieved {len(delegated_data.get('wallets', []))} delegated wallets for user {user_id}")
                    return delegated_data.get('wallets', [])
                else:
                    logger.error(f"Failed to get delegated wallets: {response.status_code} - {response.text}")
                    return []

        except Exception as e:
            logger.error(f"Error getting delegated wallets: {e}")
            return []

    async def validate_delegated_action(
        self,
        user_id: str,
        wallet_address: str,
        action_data: Dict[str, Any]
    ) -> bool:
        """
        Validate that a delegated action is permitted for the user's wallet.

        Args:
            user_id: Privy user ID
            wallet_address: Wallet address to validate
            action_data: Action details (transaction type, value, etc.)

        Returns:
            True if action is permitted, False otherwise
        """
        try:
            # Get delegated wallets for user
            delegated_wallets = await self.get_delegated_wallets(user_id)

            # Check if wallet address is in delegated wallets
            wallet_found = False
            for wallet in delegated_wallets:
                if wallet.get('address', '').lower() == wallet_address.lower():
                    wallet_found = True
                    break

            if not wallet_found:
                logger.warning(f"Wallet {wallet_address} not found in delegated wallets for user {user_id}")
                return False

            # Additional validation can be added here based on Privy's policies
            # For now, if wallet is delegated, allow the action
            logger.debug(f"Validated delegated action for wallet {wallet_address}")
            return True

        except Exception as e:
            logger.error(f"Error validating delegated action: {e}")
            return False

    async def execute_delegated_transaction(
        self,
        user_id: str,
        wallet_address: str,
        transaction_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Execute a transaction using delegated wallet authority via Privy.

        Args:
            user_id: Privy user ID
            wallet_address: Delegated wallet address
            transaction_data: Transaction details

        Returns:
            Transaction result or None if failed
        """
        try:
            # Validate delegated action first
            if not await self.validate_delegated_action(user_id, wallet_address, transaction_data):
                logger.error(f"Delegated action validation failed for user {user_id}")
                return None

            # Prepare transaction request for Privy
            headers = {
                'Authorization': f'Bearer {self.privy_app_secret}',
                'privy-app-id': self.privy_app_id,
                'Content-Type': 'application/json'
            }

            # Format transaction data for Privy API
            privy_tx_data = {
                'wallet_address': wallet_address,
                'chain_type': 'ethereum',  # or 'solana' depending on wallet
                'transaction': {
                    'to': transaction_data.get('to'),
                    'value': transaction_data.get('value', '0'),
                    'data': transaction_data.get('data', '0x'),
                    'gas_limit': transaction_data.get('gas_limit'),
                    'gas_price': transaction_data.get('gas_price')
                }
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.privy_api_url}/v1/users/{user_id}/delegated_wallets/execute",
                    headers=headers,
                    json=privy_tx_data
                )

                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"Successfully executed delegated transaction: {result.get('transaction_hash')}")
                    return result
                else:
                    logger.error(f"Failed to execute delegated transaction: {response.status_code} - {response.text}")
                    return None

        except Exception as e:
            logger.error(f"Error executing delegated transaction: {e}")
            return None

    async def setup_agent_delegation(
        self,
        user_id: str,
        agent_type: str,
        wallet_address: str
    ) -> bool:
        """
        Set up wallet delegation for an AI agent.

        This method handles the server-side setup after the user has
        delegated their wallet via the frontend React component.

        Args:
            user_id: Privy user ID
            agent_type: Type of agent (sakura, ryu, yuki)
            wallet_address: User's wallet address to delegate

        Returns:
            True if delegation setup successful
        """
        try:
            # Verify the wallet is delegated
            delegated_wallets = await self.get_delegated_wallets(user_id)

            wallet_delegated = any(
                wallet.get('address', '').lower() == wallet_address.lower()
                for wallet in delegated_wallets
            )

            if not wallet_delegated:
                logger.error(f"Wallet {wallet_address} is not delegated for user {user_id}")
                return False

            # Store delegation info for agent
            delegation_info = {
                'user_id': user_id,
                'agent_type': agent_type,
                'wallet_address': wallet_address,
                'delegated_at': datetime.now().isoformat(),
                'status': 'active'
            }

            # Store in local cache and database
            cache_key = f"{user_id}_{agent_type}"
            self.delegated_wallets[cache_key] = delegation_info

            logger.info(f"Set up agent delegation: {agent_type} for user {user_id} with wallet {wallet_address}")
            return True

        except Exception as e:
            logger.error(f"Error setting up agent delegation: {e}")
            return False

    def has_delegation(self, user_id: str, agent_type: str) -> bool:
        """
        Check if user has active delegation for an agent.

        Args:
            user_id: Privy user ID
            agent_type: Agent type

        Returns:
            True if delegation exists and is active
        """
        try:
            cache_key = f"{user_id}_{agent_type}"
            delegation = self.delegated_wallets.get(cache_key)

            return delegation is not None and delegation.get('status') == 'active'

        except Exception as e:
            logger.error(f"Error checking delegation: {e}")
            return False


# Global service instance
privy_auth_service = None


def get_privy_auth_service() -> PrivyAuthService:
    """Get global Privy authentication service instance."""
    global privy_auth_service
    
    if privy_auth_service is None:
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        privy_app_id = os.getenv('PRIVY_APP_ID')
        privy_app_secret = os.getenv('PRIVY_APP_SECRET')
        testnet = os.getenv('HYPERLIQUID_TESTNET', 'true').lower() == 'true'
        
        if not privy_app_id or not privy_app_secret:
            raise ValueError("PRIVY_APP_ID and PRIVY_APP_SECRET must be configured")
        
        privy_auth_service = PrivyAuthService(privy_app_id, privy_app_secret, testnet)
    
    return privy_auth_service
