"""
eth_account-compatible signer backed by Privy's delegated wallet RPC.

The Hyperliquid SDK signs actions with wallet.sign_message(structured_data),
where structured_data is an EIP-712 SignableMessage. This account computes the
EIP-712 digest locally and asks Privy to sign the raw hash (secp256k1_sign)
with the user's embedded wallet, authorized by the platform's delegated
session-signer key - the same signing path the funding pipeline uses in
production.

No private key ever touches this process.
"""

import base64
import logging
import os
from typing import Any, Dict, Optional

import httpx
from eth_account.messages import SignableMessage, _hash_eip191_message, encode_typed_data
from hexbytes import HexBytes
from privy.lib.authorization_signatures import get_authorization_signature

logger = logging.getLogger(__name__)

PRIVY_API_ROOT = "https://api.privy.io"


class PrivyDelegatedAccount:
    """Minimal LocalAccount stand-in for the Hyperliquid SDK."""

    def __init__(self, wallet_id: str, address: str):
        self.wallet_id = wallet_id
        self.address = address

    def sign_message(self, signable_message: SignableMessage) -> Dict[str, int]:
        """Sign an EIP-712 message digest through Privy's wallet RPC."""
        digest = _hash_eip191_message(signable_message)
        result = self._wallet_rpc(
            {
                "method": "secp256k1_sign",
                "params": {"hash": "0x" + digest.hex()},
            }
        )

        data = result.get("data") or {}
        signature = HexBytes(data.get("signature") or "")
        if len(signature) != 65:
            raise ValueError(f"Privy returned an unexpected signature payload: {result}")

        r = int.from_bytes(signature[0:32], "big")
        s = int.from_bytes(signature[32:64], "big")
        v = signature[64]
        if v < 27:
            v += 27

        # The Hyperliquid SDK reads r/s/v by key from the result.
        return {"r": r, "s": s, "v": v}

    def sign_typed_data(
        self,
        domain_data: Optional[Dict[str, Any]] = None,
        message_types: Optional[Dict[str, Any]] = None,
        message_data: Optional[Dict[str, Any]] = None,
        full_message: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, int]:
        """Support the typed-data signer interface used by newer Hyperliquid SDKs."""
        return self.sign_message(
            encode_typed_data(
                domain_data=domain_data,
                message_types=message_types,
                message_data=message_data,
                full_message=full_message,
            )
        )

    def _wallet_rpc(self, body: Dict[str, Any]) -> Dict[str, Any]:
        privy_app_id = os.getenv("PRIVY_APP_ID")
        privy_app_secret = os.getenv("PRIVY_APP_SECRET")
        auth_key_private = os.getenv("PRIVY_AUTHORIZATION_KEY_PRIVATE_KEY")
        if not all([privy_app_id, privy_app_secret, auth_key_private]):
            raise ValueError("Missing Privy delegation credentials for wallet signing")

        url = f"{PRIVY_API_ROOT}/v1/wallets/{self.wallet_id}/rpc"
        signature = get_authorization_signature(
            url=url,
            body=body,
            method="POST",
            app_id=privy_app_id,
            private_key=auth_key_private,
        )
        basic_auth = base64.b64encode(f"{privy_app_id}:{privy_app_secret}".encode()).decode()

        response = httpx.post(
            url,
            headers={
                "Authorization": f"Basic {basic_auth}",
                "privy-app-id": privy_app_id,
                "privy-authorization-signature": signature,
                "Content-Type": "application/json",
            },
            json=body,
            timeout=30.0,
        )
        if response.status_code != 200:
            raise ValueError(f"Privy signing RPC error {response.status_code}: {response.text}")
        return response.json()
