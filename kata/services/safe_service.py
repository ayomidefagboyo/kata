"""
Safe service for managing Gnosis Safe contracts on Base for Team Agents.
"""
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
import httpx
from eth_account import Account
from eth_typing import HexStr
from web3 import Web3
# Skip geth_poa_middleware for now - not needed for this implementation
# from web3.middleware import geth_poa_middleware

from kata.config.settings import settings

logger = logging.getLogger(__name__)


class SafeService:
    """Service for managing Gnosis Safe contracts on Base."""

    def __init__(self):
        """Initialize the Safe service."""
        # Use configuration from settings
        self.base_rpc_url = settings.BASE_RPC_URL
        self.safe_api_url = settings.SAFE_CORE_SDK_API_URL
        self.relay_service_url = settings.SAFE_RELAY_SERVICE_URL
        self.chain_id = settings.CHAIN_ID

        # Initialize Web3
        self.w3 = Web3(Web3.HTTPProvider(self.base_rpc_url))
        # Skip POA middleware injection for now

        # Safe Master Copy addresses on Base Sepolia testnet
        self.safe_master_copy = "0x3E5c63644E683549055b9Be8653de26E0B4CD36E"  # Safe 1.3.0 on Base Sepolia
        self.safe_proxy_factory = "0xa6B71E26C5e0845f74c812102Ca7114b6a896AB2"  # Safe Proxy Factory on Base Sepolia

        self.session = None

    async def _get_session(self) -> httpx.AsyncClient:
        """Get or create HTTP session."""
        if self.session is None:
            self.session = httpx.AsyncClient(timeout=30.0)
        return self.session

    async def close(self):
        """Close HTTP session."""
        if self.session:
            await self.session.aclose()
            self.session = None

    async def deploy_safe(
        self,
        owners: List[str],
        threshold: int,
        salt_nonce: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Deploy a new Gnosis Safe contract on Base.

        Args:
            owners: List of owner addresses
            threshold: Number of required confirmations
            salt_nonce: Optional salt for deterministic address generation

        Returns:
            Dict containing safe address and deployment transaction info
        """
        try:
            # Validate inputs
            if len(owners) < threshold:
                raise ValueError("Threshold cannot exceed number of owners")

            if threshold < 1:
                raise ValueError("Threshold must be at least 1")

            # Convert addresses to checksummed format
            owners = [Web3.to_checksum_address(owner) for owner in owners]

            # Generate salt nonce if not provided
            if salt_nonce is None:
                salt_nonce = int(asyncio.get_event_loop().time())

            # Predict Safe address
            safe_address = await self._predict_safe_address(owners, threshold, salt_nonce)

            # Check if Safe already exists at this address
            if await self._is_contract_deployed(safe_address):
                logger.info(f"Safe already deployed at {safe_address}")
                return {
                    "safe_address": safe_address,
                    "transaction_hash": None,
                    "already_deployed": True
                }

            # Deploy Safe using relay service (if available) or return deployment params
            deployment_data = await self._prepare_safe_deployment(owners, threshold, salt_nonce)

            return {
                "safe_address": safe_address,
                "deployment_data": deployment_data,
                "owners": owners,
                "threshold": threshold,
                "salt_nonce": salt_nonce,
                "chain_id": self.chain_id
            }

        except Exception as e:
            logger.error(f"Error deploying Safe: {e}")
            raise

    async def _predict_safe_address(
        self,
        owners: List[str],
        threshold: int,
        salt_nonce: int
    ) -> str:
        """Predict the Safe address before deployment."""
        try:
            # This is a simplified prediction - in production, you'd use the actual Safe SDK
            # For now, we'll generate a deterministic address based on the inputs
            import hashlib

            # Create deterministic seed from owners, threshold, and nonce
            seed_data = f"{''.join(sorted(owners))}-{threshold}-{salt_nonce}"
            seed_hash = hashlib.sha256(seed_data.encode()).hexdigest()

            # Generate a fake address for demonstration (in production, use actual Safe SDK)
            predicted_address = f"0x{seed_hash[:40]}"
            return Web3.to_checksum_address(predicted_address)

        except Exception as e:
            logger.error(f"Error predicting Safe address: {e}")
            raise

    async def _is_contract_deployed(self, address: str) -> bool:
        """Check if a contract is deployed at the given address."""
        try:
            code = self.w3.eth.get_code(Web3.to_checksum_address(address))
            return len(code) > 0
        except Exception as e:
            logger.error(f"Error checking contract deployment: {e}")
            return False

    async def _prepare_safe_deployment(
        self,
        owners: List[str],
        threshold: int,
        salt_nonce: int
    ) -> Dict[str, Any]:
        """Prepare Safe deployment transaction data."""
        try:
            # Safe setup data
            setup_data = {
                "owners": owners,
                "threshold": threshold,
                "to": "0x0000000000000000000000000000000000000000",  # No delegate call
                "data": "0x",  # No setup data
                "fallback_handler": "0xf48f2B2d2a534e402487b3ee7C18c33Aec0Fe5e4",  # Default fallback handler
                "payment_token": "0x0000000000000000000000000000000000000000",  # ETH
                "payment": 0,  # No payment
                "payment_receiver": "0x0000000000000000000000000000000000000000"
            }

            return {
                "setup_data": setup_data,
                "master_copy": self.safe_master_copy,
                "proxy_factory": self.safe_proxy_factory,
                "salt_nonce": salt_nonce,
                "chain_id": self.chain_id
            }

        except Exception as e:
            logger.error(f"Error preparing Safe deployment: {e}")
            raise

    async def get_safe_info(self, safe_address: str) -> Dict[str, Any]:
        """Get Safe contract information."""
        try:
            session = await self._get_session()

            url = f"{self.safe_api_url}/v1/safes/{safe_address}/"
            response = await session.get(url)

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Error getting Safe info: {response.status_code}")
                return {}

        except Exception as e:
            logger.error(f"Error getting Safe info: {e}")
            return {}

    async def get_safe_owners(self, safe_address: str) -> List[str]:
        """Get Safe owners."""
        try:
            safe_info = await self.get_safe_info(safe_address)
            return safe_info.get("owners", [])
        except Exception as e:
            logger.error(f"Error getting Safe owners: {e}")
            return []

    async def get_safe_threshold(self, safe_address: str) -> int:
        """Get Safe threshold."""
        try:
            safe_info = await self.get_safe_info(safe_address)
            return safe_info.get("threshold", 0)
        except Exception as e:
            logger.error(f"Error getting Safe threshold: {e}")
            return 0

    async def get_safe_balance(self, safe_address: str, token_address: Optional[str] = None) -> float:
        """Get Safe balance for ETH or specific token."""
        try:
            if token_address is None:
                # Get ETH balance
                balance_wei = self.w3.eth.get_balance(Web3.to_checksum_address(safe_address))
                return float(self.w3.from_wei(balance_wei, 'ether'))
            else:
                # Get token balance (would need ERC20 ABI implementation)
                logger.warning("Token balance checking not implemented yet")
                return 0.0

        except Exception as e:
            logger.error(f"Error getting Safe balance: {e}")
            return 0.0

    async def propose_transaction(
        self,
        safe_address: str,
        to: str,
        value: int,
        data: str,
        operation: int = 0,
        safe_tx_gas: int = 0,
        base_gas: int = 0,
        gas_price: int = 0,
        gas_token: str = "0x0000000000000000000000000000000000000000",
        refund_receiver: str = "0x0000000000000000000000000000000000000000",
        nonce: Optional[int] = None
    ) -> Dict[str, Any]:
        """Propose a new transaction to the Safe."""
        try:
            session = await self._get_session()

            # Get next nonce if not provided
            if nonce is None:
                nonce = await self._get_next_nonce(safe_address)

            # Prepare transaction data
            transaction_data = {
                "to": to,
                "value": str(value),
                "data": data,
                "operation": operation,
                "safeTxGas": str(safe_tx_gas),
                "baseGas": str(base_gas),
                "gasPrice": str(gas_price),
                "gasToken": gas_token,
                "refundReceiver": refund_receiver,
                "nonce": nonce
            }

            url = f"{self.safe_api_url}/v1/safes/{safe_address}/multisig-transactions/"
            response = await session.post(url, json=transaction_data)

            if response.status_code == 201:
                return response.json()
            else:
                logger.error(f"Error proposing transaction: {response.status_code} - {response.text}")
                raise Exception(f"Failed to propose transaction: {response.text}")

        except Exception as e:
            logger.error(f"Error proposing transaction: {e}")
            raise

    async def _get_next_nonce(self, safe_address: str) -> int:
        """Get the next nonce for the Safe."""
        try:
            session = await self._get_session()

            url = f"{self.safe_api_url}/v1/safes/{safe_address}/"
            response = await session.get(url)

            if response.status_code == 200:
                safe_info = response.json()
                return safe_info.get("nonce", 0)
            else:
                logger.error(f"Error getting Safe nonce: {response.status_code}")
                return 0

        except Exception as e:
            logger.error(f"Error getting Safe nonce: {e}")
            return 0

    async def confirm_transaction(
        self,
        safe_address: str,
        safe_tx_hash: str,
        signature: str
    ) -> Dict[str, Any]:
        """Confirm a pending transaction."""
        try:
            session = await self._get_session()

            confirmation_data = {
                "signature": signature
            }

            url = f"{self.safe_api_url}/v1/safes/{safe_address}/multisig-transactions/{safe_tx_hash}/confirmations/"
            response = await session.post(url, json=confirmation_data)

            if response.status_code == 201:
                return response.json()
            else:
                logger.error(f"Error confirming transaction: {response.status_code} - {response.text}")
                raise Exception(f"Failed to confirm transaction: {response.text}")

        except Exception as e:
            logger.error(f"Error confirming transaction: {e}")
            raise

    async def execute_transaction(
        self,
        safe_address: str,
        safe_tx_hash: str
    ) -> Dict[str, Any]:
        """Execute a fully confirmed transaction."""
        try:
            session = await self._get_session()

            url = f"{self.safe_api_url}/v1/safes/{safe_address}/multisig-transactions/{safe_tx_hash}/"
            response = await session.get(url)

            if response.status_code == 200:
                transaction = response.json()

                # Check if transaction has enough confirmations
                confirmations = transaction.get("confirmations", [])
                threshold = await self.get_safe_threshold(safe_address)

                if len(confirmations) >= threshold:
                    # Transaction can be executed
                    # In a real implementation, you'd execute the transaction on-chain
                    logger.info(f"Transaction {safe_tx_hash} ready for execution")
                    return {
                        "status": "ready_for_execution",
                        "confirmations": len(confirmations),
                        "threshold": threshold,
                        "transaction": transaction
                    }
                else:
                    return {
                        "status": "insufficient_confirmations",
                        "confirmations": len(confirmations),
                        "threshold": threshold,
                        "required": threshold - len(confirmations)
                    }
            else:
                logger.error(f"Error getting transaction: {response.status_code}")
                raise Exception(f"Failed to get transaction: {response.text}")

        except Exception as e:
            logger.error(f"Error executing transaction: {e}")
            raise

    async def get_pending_transactions(self, safe_address: str) -> List[Dict[str, Any]]:
        """Get pending transactions for a Safe."""
        try:
            session = await self._get_session()

            url = f"{self.safe_api_url}/v1/safes/{safe_address}/multisig-transactions/"
            params = {"executed": "false"}
            response = await session.get(url, params=params)

            if response.status_code == 200:
                data = response.json()
                return data.get("results", [])
            else:
                logger.error(f"Error getting pending transactions: {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"Error getting pending transactions: {e}")
            return []


# Global Safe service instance
safe_service = SafeService()