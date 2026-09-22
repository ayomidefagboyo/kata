"""
Privy Wallet Signing Service

This service handles transaction signing through Privy's official SDK and remote wallet infrastructure.
Used for signing bridge transactions without exposing private keys to the backend.
"""

import asyncio
import base64
import logging
import os
from typing import Dict, Any, Optional
import httpx
import json
from dataclasses import dataclass
from privy.lib.authorization_signatures import get_authorization_signature

from privy import PrivyAPI
from privy.types import User, Wallet

logger = logging.getLogger(__name__)


@dataclass
class SigningRequest:
    """Signing request data structure."""
    user_id: str
    chain_id: int
    transaction_data: Dict[str, Any]
    signing_method: str = "privy_remote"
    

@dataclass
class SigningResult:
    """Signing result data structure."""
    success: bool
    signed_transaction: Optional[str] = None
    transaction_hash: Optional[str] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class PrivySigningService:
    """
    Service for handling transaction signing through Privy wallets.
    
    This service integrates with Privy's remote wallet signing capabilities
    to sign transactions without exposing private keys to the backend.
    """
    
    def __init__(self, privy_app_id: str, privy_app_secret: str, testnet: bool = False):
        """Initialize Privy signing service."""
        self.privy_app_id = privy_app_id
        self.privy_app_secret = privy_app_secret
        self.testnet = testnet

        # Initialize official Privy client
        self.privy_client = PrivyAPI(
            app_id=privy_app_id,
            app_secret=privy_app_secret
        )

        # HTTP client for additional API calls if needed
        self.http_client = httpx.AsyncClient(
            timeout=30.0,
            headers={
                "Authorization": f"Bearer {privy_app_secret}",
                "Content-Type": "application/json"
            }
        )

        logger.info(f"Privy signing service initialized with official SDK (testnet: {testnet})")
    
    async def sign_transaction(
        self, 
        user_id: str,
        chain_id: int,
        transaction_data: Dict[str, Any],
        wallet_address: str
    ) -> SigningResult:
        """
        Sign a transaction using Privy remote wallet.
        
        Args:
            user_id: User ID in our system
            chain_id: Blockchain chain ID (e.g., 8453 for Base)
            transaction_data: Raw transaction data to sign
            wallet_address: User's wallet address
            
        Returns:
            SigningResult with signed transaction or error
        """
        try:
            logger.info(f"Initiating Privy signing for user {user_id} on chain {chain_id}")
            
            # Use official Privy SDK for transaction signing:
            # 1. Look up user's Privy wallet ID from their user_id
            # 2. Call Privy's signing API to sign the transaction
            # 3. Return the signed transaction for broadcasting

            signing_request = SigningRequest(
                user_id=user_id,
                chain_id=chain_id,
                transaction_data=transaction_data,
                signing_method="privy_sdk"
            )

            # Use real Privy SDK signing
            signed_result = await self._sign_with_privy_sdk(signing_request, wallet_address)
            
            return signed_result
            
        except Exception as e:
            logger.error(f"Error signing transaction with Privy: {e}")
            return SigningResult(
                success=False,
                error=str(e)
            )

    async def sign_solana_transaction(
        self,
        user_id: str,
        transaction_base64: str,
        wallet_address: str,
        reference_id: Optional[str] = None,
    ) -> SigningResult:
        """Sign and broadcast a serialized Solana transaction.

        The user's delegated Solana wallet pays the network fee from its SOL
        balance. ``sponsor`` is deliberately false: Kata never subsidizes it.
        """
        try:
            from kata.services.delegation_service import get_delegation_service

            delegation = await get_delegation_service().get_user_delegation(
                user_id,
                chain_type="solana",
            )
            if not delegation or not delegation.is_active:
                return SigningResult(
                    success=False,
                    error="No active Solana wallet delegation found",
                )
            if delegation.wallet_address != wallet_address:
                return SigningResult(
                    success=False,
                    error="Delegated Solana wallet does not match the Ryu wallet",
                )

            body: Dict[str, Any] = {
                "method": "signAndSendTransaction",
                "caip2": "solana:5eykt4UsFv8P8NJdTREpY1vzqKqZKvdp",
                "sponsor": False,
                "params": {
                    "transaction": transaction_base64,
                    "encoding": "base64",
                },
            }
            if reference_id:
                body["reference_id"] = reference_id[:64]

            result = await self._wallet_rpc(delegation.wallet_id, body)
            data = result.get("data") or {}
            tx_hash = data.get("hash") or result.get("result")
            if not tx_hash:
                raise ValueError(f"Privy returned no Solana transaction hash: {result}")
            return SigningResult(
                success=True,
                transaction_hash=tx_hash,
                metadata={
                    "signing_method": "privy_wallet_rpc",
                    "chain": "solana",
                    "wallet_address": wallet_address,
                    "wallet_id": delegation.wallet_id,
                    "transaction_id": data.get("transaction_id"),
                    "sponsored": False,
                },
            )
        except Exception as e:
            logger.error(f"Error signing Solana transaction with Privy: {e}")
            return SigningResult(success=False, error=str(e))

    async def sign_and_send_delegated_evm_transaction(
        self,
        user_id: str,
        chain_id: int,
        transaction_data: Dict[str, Any],
        wallet_address: str,
        reference_id: Optional[str] = None,
    ) -> SigningResult:
        """Send a delegated EVM transaction with gas sponsorship disabled."""
        try:
            from kata.services.delegation_service import get_delegation_service

            delegation = await get_delegation_service().get_user_delegation(
                user_id,
                chain_type="ethereum",
            )
            if not delegation or not delegation.is_active:
                return SigningResult(
                    success=False,
                    error="No active Ethereum wallet delegation found",
                )
            if delegation.wallet_address.lower() != wallet_address.lower():
                return SigningResult(
                    success=False,
                    error="Delegated Ethereum wallet does not match the Ryu wallet",
                )

            raw_value = transaction_data.get("value") or "0"
            value = (
                raw_value
                if isinstance(raw_value, str) and raw_value.startswith("0x")
                else hex(int(raw_value))
            )
            body: Dict[str, Any] = {
                "method": "eth_sendTransaction",
                "caip2": f"eip155:{chain_id}",
                "sponsor": False,
                "params": {
                    "transaction": {
                        "to": transaction_data.get("to"),
                        "value": value,
                        "data": transaction_data.get("data") or "0x",
                    },
                },
            }
            if reference_id:
                body["reference_id"] = reference_id[:64]

            result = await self._wallet_rpc(delegation.wallet_id, body)
            data = result.get("data") or {}
            tx_hash = data.get("hash") or result.get("result")
            if not tx_hash:
                raise ValueError(f"Privy returned no EVM transaction hash: {result}")
            return SigningResult(
                success=True,
                transaction_hash=tx_hash,
                metadata={
                    "signing_method": "privy_wallet_rpc",
                    "chain_id": chain_id,
                    "wallet_address": wallet_address,
                    "wallet_id": delegation.wallet_id,
                    "transaction_id": data.get("transaction_id"),
                    "sponsored": False,
                },
            )
        except Exception as e:
            logger.error(f"Error signing delegated EVM transaction with Privy: {e}")
            return SigningResult(success=False, error=str(e))

    async def _wallet_rpc(
        self,
        wallet_id: str,
        body: Dict[str, Any],
    ) -> Dict[str, Any]:
        auth_key_private = os.getenv("PRIVY_AUTHORIZATION_KEY_PRIVATE_KEY")
        if not all([
            self.privy_app_id,
            self.privy_app_secret,
            auth_key_private,
        ]):
            raise ValueError("Missing Privy delegation credentials")

        url = f"https://api.privy.io/v1/wallets/{wallet_id}/rpc"
        signature = get_authorization_signature(
            url=url,
            body=body,
            method="POST",
            app_id=self.privy_app_id,
            private_key=auth_key_private,
        )
        basic_auth = base64.b64encode(
            f"{self.privy_app_id}:{self.privy_app_secret}".encode()
        ).decode()
        response = await self.http_client.post(
            url,
            headers={
                "Authorization": f"Basic {basic_auth}",
                "privy-app-id": self.privy_app_id,
                "privy-authorization-signature": signature,
                "Content-Type": "application/json",
            },
            json=body,
        )
        if response.status_code != 200:
            raise ValueError(
                f"Privy wallet RPC error {response.status_code}: {response.text}"
            )
        return response.json()
    
    async def _sign_with_privy_sdk(
        self,
        signing_request: SigningRequest,
        wallet_address: str
    ) -> SigningResult:
        """Sign transaction using official Privy SDK."""
        try:
            # Use official Privy SDK for transaction signing
            #
            # 1. Call Privy SDK to sign the transaction
            # 2. Handle the response and return signed transaction

            logger.info(f"Signing transaction with Privy SDK for wallet {wallet_address}")

            # Call Privy SDK signing method
            signing_result = await asyncio.to_thread(
                self.privy_client.wallets.send_transaction,
                user_id=signing_request.user_id,
                wallet_address=wallet_address,
                transaction={
                    "to": signing_request.transaction_data.get("to"),
                    "value": signing_request.transaction_data.get("value", "0"),
                    "data": signing_request.transaction_data.get("data", "0x"),
                    "chainId": signing_request.chain_id
                }
            )

            if signing_result and hasattr(signing_result, 'hash'):
                logger.info(f"Privy SDK signing completed for wallet {wallet_address}: {signing_result.hash}")

                return SigningResult(
                    success=True,
                    signed_transaction=getattr(signing_result, 'raw_transaction', None),
                    transaction_hash=signing_result.hash,
                    metadata={
                        "signing_method": "privy_sdk",
                        "chain_id": signing_request.chain_id,
                        "wallet_address": wallet_address,
                        "block_number": getattr(signing_result, 'block_number', None),
                        "gas_used": getattr(signing_result, 'gas_used', None)
                    }
                )
            else:
                raise Exception("Failed to get transaction hash from Privy SDK")

        except Exception as e:
            logger.error(f"Error in Privy SDK signing: {e}")
            return SigningResult(
                success=False,
                error=str(e)
            )
    
    async def get_user_wallet_address(self, user_id: str) -> Optional[str]:
        """
        Get user's Privy wallet address.
        
        Args:
            user_id: User ID in our system
            
        Returns:
            Wallet address or None if not found
        """
        try:
            # Use official Privy SDK to get user wallet address
            # 1. Look up user's Privy user ID from our user_id
            # 2. Call Privy API to get their wallet address
            # 3. Return the address

            try:
                user_data = await asyncio.to_thread(
                    self.privy_client.users.get,
                    user_id
                )

                if user_data and user_data.linked_accounts:
                    for account in user_data.linked_accounts:
                        if (hasattr(account, 'type') and account.type == 'wallet' and
                            hasattr(account, 'chain_type') and account.chain_type == 'ethereum'):
                            logger.info(f"Retrieved wallet address for user {user_id}: {account.address}")
                            return account.address

                logger.warning(f"No Ethereum wallet found for user {user_id}")
                return None

            except Exception as e:
                logger.error(f"Error getting user wallet address via SDK: {e}")
                return None
            
        except Exception as e:
            logger.error(f"Error getting user wallet address: {e}")
            return None
    
    async def verify_wallet_ownership(self, user_id: str, wallet_address: str) -> bool:
        """
        Verify that the user owns the specified wallet.
        
        Args:
            user_id: User ID in our system
            wallet_address: Wallet address to verify
            
        Returns:
            True if user owns the wallet, False otherwise
        """
        try:
            # Use official Privy SDK to verify wallet ownership
            try:
                user_data = await asyncio.to_thread(
                    self.privy_client.users.get,
                    user_id
                )

                if user_data and user_data.linked_accounts:
                    for account in user_data.linked_accounts:
                        if (hasattr(account, 'type') and account.type == 'wallet' and
                            hasattr(account, 'address') and
                            account.address.lower() == wallet_address.lower()):
                            logger.info(f"Verified wallet ownership for user {user_id}: {wallet_address}")
                            return True

                logger.warning(f"Wallet {wallet_address} not found for user {user_id}")
                return False

            except Exception as e:
                logger.error(f"Error verifying wallet ownership via SDK: {e}")
                return False
            
        except Exception as e:
            logger.error(f"Error verifying wallet ownership: {e}")
            return False
    
    async def estimate_gas_fees(
        self, 
        chain_id: int, 
        transaction_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Estimate gas fees for a transaction.
        
        Args:
            chain_id: Blockchain chain ID
            transaction_data: Transaction data
            
        Returns:
            Gas fee estimation
        """
        try:
            # In production, this would call chain RPC or gas estimation APIs
            # For now, provide estimates based on chain
            
            if chain_id == 8453:  # Base
                return {
                    "gas_limit": 21000,
                    "gas_price": "1000000000",  # 1 gwei
                    "estimated_fee_eth": 0.000021,
                    "estimated_fee_usd": 0.05
                }
            elif chain_id == 42161:  # Arbitrum
                return {
                    "gas_limit": 21000,
                    "gas_price": "100000000",  # 0.1 gwei
                    "estimated_fee_eth": 0.0000021,
                    "estimated_fee_usd": 0.005
                }
            else:
                return {
                    "gas_limit": 21000,
                    "gas_price": "2000000000",  # 2 gwei
                    "estimated_fee_eth": 0.000042,
                    "estimated_fee_usd": 0.10
                }
                
        except Exception as e:
            logger.error(f"Error estimating gas fees: {e}")
            return {
                "error": str(e)
            }
    
    async def close(self):
        """Close the signing service and cleanup resources."""
        try:
            await self.http_client.aclose()
            logger.info("Privy signing service closed")
        except Exception as e:
            logger.error(f"Error closing signing service: {e}")


# Global signing service instance
privy_signing_service = None


def get_privy_signing_service(
    privy_app_id: str, 
    privy_app_secret: str, 
    testnet: bool = False
) -> PrivySigningService:
    """Get or create global Privy signing service instance."""
    global privy_signing_service
    
    if privy_signing_service is None:
        privy_signing_service = PrivySigningService(
            privy_app_id=privy_app_id,
            privy_app_secret=privy_app_secret,
            testnet=testnet
        )
    
    return privy_signing_service
