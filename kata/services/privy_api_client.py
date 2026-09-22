"""
Real Privy API Client for Python Backend

This module provides a proper Privy API integration replacing the simulation code.
It implements the actual Privy REST API calls for authentication, user management,
and wallet operations.
"""

import asyncio
import logging
import json
import jwt
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import httpx
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PrivyUser:
    """Privy user data structure."""
    id: str
    created_at: str
    linked_accounts: List[Dict[str, Any]]
    mfa_methods: List[Dict[str, Any]]
    has_accepted_terms: bool
    is_guest: bool


@dataclass
class PrivyWallet:
    """Privy wallet data structure."""
    address: str
    chain_type: str
    wallet_type: str
    verified: bool
    imported: bool
    delegated: bool
    recovery_method: Optional[str] = None


class PrivyAPIClient:
    """
    Real Privy API client for authentication and wallet operations.

    This replaces the simulation code with actual Privy API integration.
    See: https://docs.privy.io/guide/server-side/users
    """

    def __init__(self, app_id: str, app_secret: str, base_url: str = "https://auth.privy.io"):
        """Initialize Privy API client."""
        self.app_id = app_id
        self.app_secret = app_secret
        self.base_url = base_url

        # HTTP client with authentication
        self.http_client = httpx.AsyncClient(
            timeout=30.0,
            headers={
                "privy-app-id": app_id,
                "Authorization": f"Bearer {app_secret}",
                "Content-Type": "application/json"
            }
        )

        logger.info(f"Privy API client initialized for app: {app_id}")

    async def verify_auth_token(self, auth_token: str) -> PrivyUser:
        """
        Verify user's authentication token and get user info.

        Args:
            auth_token: User's Privy access token

        Returns:
            PrivyUser with verified user information

        Raises:
            ValueError: If token verification fails
        """
        try:
            # First, verify the JWT token structure
            try:
                # Decode without verification to check structure
                decoded_header = jwt.get_unverified_header(auth_token)
                decoded_payload = jwt.decode(auth_token, options={"verify_signature": False})

                # Check if token has required fields
                if not decoded_payload.get("sub"):
                    raise ValueError("Token missing subject (user ID)")

            except jwt.InvalidTokenError as e:
                raise ValueError(f"Invalid token format: {e}")

            # Verify token with Privy API
            response = await self.http_client.get(
                f"{self.base_url}/api/v1/users/me",
                headers={"Authorization": f"Bearer {auth_token}"}
            )

            if response.status_code == 401:
                raise ValueError("Invalid or expired authentication token")
            elif response.status_code != 200:
                raise ValueError(f"Privy API error: {response.status_code} - {response.text}")

            user_data = response.json()

            return PrivyUser(
                id=user_data["id"],
                created_at=user_data["created_at"],
                linked_accounts=user_data.get("linked_accounts", []),
                mfa_methods=user_data.get("mfa_methods", []),
                has_accepted_terms=user_data.get("has_accepted_terms", False),
                is_guest=user_data.get("is_guest", False)
            )

        except httpx.RequestError as e:
            logger.error(f"Network error verifying token: {e}")
            raise ValueError(f"Network error: {e}")
        except Exception as e:
            logger.error(f"Error verifying auth token: {e}")
            raise ValueError(f"Token verification failed: {e}")

    async def get_user_wallets(self, user_id: str) -> List[PrivyWallet]:
        """
        Get all wallets for a user.

        Args:
            user_id: Privy user ID

        Returns:
            List of PrivyWallet objects
        """
        try:
            response = await self.http_client.get(
                f"{self.base_url}/api/v1/users/{user_id}"
            )

            if response.status_code != 200:
                logger.error(f"Failed to get user wallets: {response.status_code} - {response.text}")
                return []

            user_data = response.json()
            wallets = []

            # Extract wallets from linked accounts
            for account in user_data.get("linked_accounts", []):
                if account.get("type") == "wallet":
                    wallets.append(PrivyWallet(
                        address=account["address"],
                        chain_type=account.get("chain_type", "ethereum"),
                        wallet_type=account.get("wallet_type", "privy"),
                        verified=account.get("verified", False),
                        imported=account.get("imported", False),
                        delegated=account.get("delegated", False),
                        recovery_method=account.get("recovery_method")
                    ))

            logger.info(f"Retrieved {len(wallets)} wallets for user {user_id}")
            return wallets

        except Exception as e:
            logger.error(f"Error getting user wallets: {e}")
            return []

    async def get_primary_wallet(self, user_id: str) -> Optional[PrivyWallet]:
        """
        Get user's primary wallet address.

        Args:
            user_id: Privy user ID

        Returns:
            Primary PrivyWallet or None if not found
        """
        wallets = await self.get_user_wallets(user_id)

        if not wallets:
            return None

        # Return first Ethereum wallet (prioritize embedded wallets)
        ethereum_wallets = [w for w in wallets if w.chain_type == "ethereum"]
        if ethereum_wallets:
            # Prefer Privy embedded wallets over imported ones
            privy_wallets = [w for w in ethereum_wallets if w.wallet_type == "privy"]
            return privy_wallets[0] if privy_wallets else ethereum_wallets[0]

        # Fallback to first available wallet
        return wallets[0]

    async def request_wallet_signature(
        self,
        user_id: str,
        wallet_address: str,
        message_to_sign: str
    ) -> Dict[str, Any]:
        """
        Request wallet signature through Privy.

        Args:
            user_id: Privy user ID
            wallet_address: Wallet address to sign with
            message_to_sign: Message to sign

        Returns:
            Signature result
        """
        try:
            payload = {
                "message": message_to_sign,
                "address": wallet_address
            }

            response = await self.http_client.post(
                f"{self.base_url}/api/v1/users/{user_id}/wallets/sign",
                json=payload
            )

            if response.status_code != 200:
                logger.error(f"Signature request failed: {response.status_code} - {response.text}")
                return {"success": False, "error": "Signature request failed"}

            result = response.json()
            logger.info(f"Signature requested for wallet {wallet_address}")

            return {
                "success": True,
                "signature": result.get("signature"),
                "message": message_to_sign,
                "address": wallet_address
            }

        except Exception as e:
            logger.error(f"Error requesting wallet signature: {e}")
            return {"success": False, "error": str(e)}

    async def send_transaction(
        self,
        user_id: str,
        wallet_address: str,
        transaction_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Send transaction through Privy embedded wallet.

        Args:
            user_id: Privy user ID
            wallet_address: Wallet address to send from
            transaction_data: Transaction details

        Returns:
            Transaction result
        """
        try:
            payload = {
                "from": wallet_address,
                "to": transaction_data.get("to"),
                "value": transaction_data.get("value", "0"),
                "data": transaction_data.get("data", "0x"),
                "gasLimit": transaction_data.get("gasLimit"),
                "gasPrice": transaction_data.get("gasPrice"),
                "chainId": transaction_data.get("chainId", 1)
            }

            response = await self.http_client.post(
                f"{self.base_url}/api/v1/users/{user_id}/wallets/send_transaction",
                json=payload
            )

            if response.status_code != 200:
                logger.error(f"Transaction failed: {response.status_code} - {response.text}")
                return {"success": False, "error": "Transaction failed"}

            result = response.json()
            logger.info(f"Transaction sent: {result.get('hash')}")

            return {
                "success": True,
                "hash": result.get("hash"),
                "from": wallet_address,
                "to": transaction_data.get("to"),
                "value": transaction_data.get("value"),
                "gasUsed": result.get("gasUsed"),
                "blockNumber": result.get("blockNumber")
            }

        except Exception as e:
            logger.error(f"Error sending transaction: {e}")
            return {"success": False, "error": str(e)}

    async def get_transaction_status(self, transaction_hash: str, chain_id: int = 1) -> Dict[str, Any]:
        """
        Get transaction status.

        Args:
            transaction_hash: Transaction hash to check
            chain_id: Blockchain chain ID

        Returns:
            Transaction status information
        """
        try:
            response = await self.http_client.get(
                f"{self.base_url}/api/v1/transactions/{transaction_hash}",
                params={"chainId": chain_id}
            )

            if response.status_code != 200:
                return {"status": "unknown", "error": "Failed to get status"}

            result = response.json()
            return {
                "status": result.get("status", "unknown"),
                "confirmations": result.get("confirmations", 0),
                "blockNumber": result.get("blockNumber"),
                "gasUsed": result.get("gasUsed")
            }

        except Exception as e:
            logger.error(f"Error getting transaction status: {e}")
            return {"status": "error", "error": str(e)}

    async def close(self):
        """Close the HTTP client and cleanup resources."""
        try:
            await self.http_client.aclose()
            logger.info("Privy API client closed")
        except Exception as e:
            logger.error(f"Error closing Privy API client: {e}")


# Global client instance
_privy_client: Optional[PrivyAPIClient] = None


def get_privy_client(app_id: str, app_secret: str) -> PrivyAPIClient:
    """Get or create global Privy API client instance."""
    global _privy_client

    if _privy_client is None:
        _privy_client = PrivyAPIClient(app_id, app_secret)

    return _privy_client


async def close_privy_client():
    """Close global Privy API client."""
    global _privy_client

    if _privy_client:
        await _privy_client.close()
        _privy_client = None