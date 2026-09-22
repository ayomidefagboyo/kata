"""
Yuki Funding Execution Service

Executes the real ETH -> Arbitrum USDC -> Hyperliquid deposit route that the
/funding-plan/yuki/eth endpoint previews. All transactions are sent from the
user's own Privy embedded wallet via Privy's session-signer RPC, authorized by
the platform authorization key the user delegated to.

Steps per execution:
1. convert       - swap/bridge ETH into Arbitrum USDC via the live LiFi route
2. arrival       - wait for Arbitrum USDC to land in the user's Kata wallet
3. deposit       - transfer Arbitrum USDC to the Hyperliquid bridge
4. credit        - wait for Hyperliquid to credit the account balance

Safety rails:
- Requires an active wallet delegation (user consent + policy record).
- Refuses to run unless YUKI_FUNDING_EXECUTION_ENABLED is set.
- Refuses fallback/estimated LiFi quotes; only live routes are executed.
- Refuses deposits below the Hyperliquid bridge minimum (deposits under the
  minimum are permanently lost by the bridge).
- Refuses to run against Hyperliquid testnet (bridge address is mainnet-only).
"""

import asyncio
import base64
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
from privy.lib.authorization_signatures import get_authorization_signature

from kata.config.settings import settings

logger = logging.getLogger(__name__)

# Native USDC on Arbitrum One (the only asset the Hyperliquid bridge accepts).
ARBITRUM_USDC_ADDRESS = "0xaf88d065e77c8cC2239327C5EDb3A432268e5831"
ARBITRUM_CHAIN_ID = 42161

# Hyperliquid mainnet bridge on Arbitrum. Deposits below this minimum are lost.
HYPERLIQUID_BRIDGE_MIN_USDC = 5.0

ERC20_TRANSFER_SELECTOR = "0xa9059cbb"
ERC20_BALANCE_OF_SELECTOR = "0x70a08231"

STEP_CONVERT = "convert"
STEP_ARRIVAL = "arrival"
STEP_DEPOSIT = "deposit"
STEP_CREDIT = "credit"

FUNDING_STEPS = [STEP_CONVERT, STEP_ARRIVAL, STEP_DEPOSIT, STEP_CREDIT]

CHAIN_CONFIG = {
    "arbitrum": {"chain_id": 42161, "rpc_env": "ARBITRUM_RPC_URL", "rpc_default": "https://arb1.arbitrum.io/rpc"},
    "base": {"chain_id": 8453, "rpc_env": "BASE_MAINNET_RPC_URL", "rpc_default": "https://mainnet.base.org"},
    "ethereum": {"chain_id": 1, "rpc_env": "ETHEREUM_RPC_URL", "rpc_default": "https://eth.llamarpc.com"},
    "optimism": {"chain_id": 10, "rpc_env": "OPTIMISM_RPC_URL", "rpc_default": "https://mainnet.optimism.io"},
}


@dataclass
class FundingStep:
    """One step of a funding execution."""
    name: str
    status: str = "pending"  # pending | running | completed | failed | skipped
    detail: str = ""
    tx_hash: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "tx_hash": self.tx_hash,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


@dataclass
class FundingExecution:
    """A single ETH -> Hyperliquid USDC funding run."""
    funding_id: str
    user_id: str
    wallet_address: str
    wallet_id: str
    source_chain: str
    eth_amount: float
    gas_reserve_eth: float
    eth_to_convert: float
    estimated_usdc: float
    status: str = "pending"  # pending | running | completed | failed
    current_step: str = STEP_CONVERT
    steps: List[FundingStep] = field(default_factory=lambda: [FundingStep(name) for name in FUNDING_STEPS])
    arrived_usdc: Optional[float] = None
    deposited_usdc: Optional[float] = None
    error: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = None

    def step(self, name: str) -> FundingStep:
        for entry in self.steps:
            if entry.name == name:
                return entry
        raise KeyError(f"Unknown funding step: {name}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "funding_id": self.funding_id,
            "user_id": self.user_id,
            "wallet_address": self.wallet_address,
            "source_chain": self.source_chain,
            "destination_chain": "arbitrum",
            "eth_amount": self.eth_amount,
            "gas_reserve_eth": self.gas_reserve_eth,
            "eth_to_convert": self.eth_to_convert,
            "estimated_usdc": self.estimated_usdc,
            "arrived_usdc": self.arrived_usdc,
            "deposited_usdc": self.deposited_usdc,
            "status": self.status,
            "current_step": self.current_step,
            "steps": [entry.to_dict() for entry in self.steps],
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
        }


class YukiFundingService:
    """Executes ETH -> Arbitrum USDC -> Hyperliquid deposits for Yuki funding."""

    def __init__(self):
        self.executions: Dict[str, FundingExecution] = {}
        self.tasks: Dict[str, asyncio.Task] = {}
        self.http_client = httpx.AsyncClient(timeout=30.0)
        self.supabase = None
        try:
            from kata.config.database import get_service_client
            self.supabase = get_service_client()
        except Exception as e:
            logger.warning(f"Yuki funding service running without database persistence: {e}")
        logger.info("Yuki funding service initialized")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start_funding(
        self,
        user_id: str,
        wallet_address: str,
        source_chain: str,
        eth_amount: float,
        gas_reserve_eth: float,
    ) -> Dict[str, Any]:
        """Validate preconditions and launch a funding execution in the background."""
        if not settings.YUKI_FUNDING_EXECUTION_ENABLED:
            return {
                "success": False,
                "error": (
                    "Automated funding execution is disabled. Set YUKI_FUNDING_EXECUTION_ENABLED=true "
                    "once Privy delegation is verified in this environment. Gas is paid from the "
                    "wallet's ETH reserve; no sponsorship is required."
                ),
            }

        if settings.HYPERLIQUID_TESTNET:
            return {
                "success": False,
                "error": "Funding execution targets the mainnet Hyperliquid bridge and cannot run with HYPERLIQUID_TESTNET=true.",
            }

        from kata.services.lifi_bridge_service import get_lifi_service
        lifi_service = get_lifi_service()
        source_chain = lifi_service.normalize_chain(source_chain)
        if source_chain not in CHAIN_CONFIG:
            return {"success": False, "error": f"Unsupported source_chain. Supported: {sorted(CHAIN_CONFIG)}"}

        if not wallet_address or not wallet_address.startswith("0x"):
            return {"success": False, "error": "A valid Kata wallet address is required"}

        # A pending run for the same wallet must finish before starting another.
        for existing in self.executions.values():
            if (
                existing.wallet_address.lower() == wallet_address.lower()
                and existing.status in {"pending", "running"}
            ):
                return {
                    "success": False,
                    "error": "A funding execution is already in progress for this wallet",
                    "funding_id": existing.funding_id,
                }

        # Delegation is the user's consent for the platform to move these funds.
        delegation = await self._get_active_delegation(user_id)
        if not delegation:
            return {"success": False, "error": "No active wallet delegation found. Complete delegation before funding."}

        if delegation.wallet_address and delegation.wallet_address.lower() != wallet_address.lower():
            return {
                "success": False,
                "error": "Delegated wallet does not match the funding wallet address",
            }

        wallet_id = delegation.wallet_id or await self._lookup_embedded_wallet_id(user_id)
        if not wallet_id:
            return {"success": False, "error": "Could not resolve the Privy wallet id for this user"}

        # The wallet itself pays source-chain gas, so never convert the full balance.
        balance_eth = await self._get_native_balance_eth(source_chain, wallet_address)
        if balance_eth is None:
            return {"success": False, "error": f"Could not read wallet ETH balance on {source_chain}"}

        # eth_amount is the NET amount to convert; the gas reserve stays in the
        # wallet on top of it. Clamp to what the balance can actually cover so
        # the reserve is never spent on the swap.
        gas_reserve = max(gas_reserve_eth, self._min_gas_reserve(source_chain))
        eth_to_convert = min(eth_amount, balance_eth - gas_reserve)
        if eth_to_convert <= 0:
            return {
                "success": False,
                "error": (
                    f"Wallet holds {balance_eth:.6f} ETH on {source_chain}; not enough to convert after the "
                    f"{gas_reserve:.6f} ETH gas reserve. Fund the wallet first."
                ),
            }

        quote = await lifi_service.get_bridge_quote(
            from_chain=source_chain,
            to_chain="arbitrum",
            from_token="ETH",
            to_token="USDC",
            amount=eth_to_convert,
            wallet_address=wallet_address,
        )
        if not quote or quote.route_data.get("fallback") or quote.route_data.get("error"):
            return {"success": False, "error": "No live LiFi route available; refusing to execute an estimated quote"}
        if not quote.route_data.get("transactionRequest"):
            return {"success": False, "error": "LiFi quote did not include an executable transaction"}

        min_deposit = max(HYPERLIQUID_BRIDGE_MIN_USDC, settings.YUKI_MIN_ALLOCATION_USDC)
        if quote.output_amount < min_deposit:
            return {
                "success": False,
                "error": (
                    f"Estimated {quote.output_amount:.2f} USDC is below the {min_deposit:.2f} USDC minimum "
                    "Hyperliquid deposit. Deposits under the bridge minimum are lost."
                ),
            }

        funding = FundingExecution(
            funding_id=str(uuid.uuid4()),
            user_id=user_id,
            wallet_address=wallet_address,
            wallet_id=wallet_id,
            source_chain=source_chain,
            eth_amount=eth_amount,
            gas_reserve_eth=gas_reserve,
            eth_to_convert=eth_to_convert,
            estimated_usdc=quote.output_amount,
        )
        self.executions[funding.funding_id] = funding
        await self._persist(funding)

        task = asyncio.create_task(self._run(funding, quote.route_data))
        self.tasks[funding.funding_id] = task
        task.add_done_callback(lambda _: self.tasks.pop(funding.funding_id, None))

        logger.info(
            f"Started Yuki funding {funding.funding_id}: {eth_to_convert:.6f} ETH on {source_chain} "
            f"-> ~{quote.output_amount:.2f} USDC for {wallet_address}"
        )
        return {"success": True, **funding.to_dict()}

    def get_status(self, funding_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Return execution status for the requesting user, or None if unknown."""
        funding = self.executions.get(funding_id)
        if funding and funding.user_id == user_id:
            return funding.to_dict()
        return self._load_persisted(funding_id, user_id)

    def get_latest_for_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Return the most recent in-memory execution for a user."""
        candidates = [entry for entry in self.executions.values() if entry.user_id == user_id]
        if not candidates:
            return None
        latest = max(candidates, key=lambda entry: entry.created_at)
        return latest.to_dict()

    # ------------------------------------------------------------------
    # Execution pipeline
    # ------------------------------------------------------------------

    async def _run(self, funding: FundingExecution, route_data: Dict[str, Any]):
        funding.status = "running"
        try:
            usdc_baseline = await self._get_usdc_balance(funding.wallet_address) or 0.0
            hl_baseline = await self._get_hyperliquid_account_value(funding.wallet_address) or 0.0

            await self._run_convert_step(funding, route_data)
            arrived = await self._run_arrival_step(funding, usdc_baseline)
            deposited = await self._run_deposit_step(funding, arrived)
            await self._run_credit_step(funding, hl_baseline, deposited)

            funding.status = "completed"
            funding.completed_at = datetime.now(timezone.utc).isoformat()
            logger.info(f"Yuki funding {funding.funding_id} completed: {deposited:.2f} USDC deposited to Hyperliquid")
        except Exception as e:
            funding.status = "failed"
            funding.error = str(e)
            step = funding.step(funding.current_step)
            if step.status == "running":
                step.status = "failed"
                step.detail = str(e)
                step.finished_at = datetime.now(timezone.utc).isoformat()
            logger.error(f"Yuki funding {funding.funding_id} failed at {funding.current_step}: {e}")
        finally:
            await self._persist(funding)

    async def _run_convert_step(self, funding: FundingExecution, route_data: Dict[str, Any]):
        step = self._begin_step(funding, STEP_CONVERT, "Submitting ETH to USDC conversion")
        await self._persist(funding)

        tx_request = route_data["transactionRequest"]
        source = CHAIN_CONFIG[funding.source_chain]
        tx_chain_id = tx_request.get("chainId")
        if tx_chain_id is not None and int(str(tx_chain_id), 0) != source["chain_id"]:
            raise ValueError(f"LiFi transaction targets chain {tx_chain_id}, expected {source['chain_id']}")

        tx_hash = await self._send_delegated_transaction(
            wallet_id=funding.wallet_id,
            chain_id=source["chain_id"],
            transaction={
                "to": tx_request["to"],
                "value": self._to_hex_quantity(tx_request.get("value", "0x0")),
                "data": tx_request.get("data", "0x"),
            },
        )
        step.tx_hash = tx_hash
        step.detail = f"Conversion submitted via {route_data.get('toolDetails', {}).get('name', 'LiFi')}"
        await self._persist(funding)

        receipt_ok = await self._wait_for_receipt(funding.source_chain, tx_hash, timeout_seconds=300)
        if not receipt_ok:
            raise ValueError(f"Conversion transaction {tx_hash} failed or was not confirmed")
        self._finish_step(step, "Conversion confirmed on-chain")
        await self._persist(funding)

    async def _run_arrival_step(self, funding: FundingExecution, usdc_baseline: float) -> float:
        step = self._begin_step(funding, STEP_ARRIVAL, "Waiting for Arbitrum USDC to arrive")
        await self._persist(funding)

        same_chain = funding.source_chain == "arbitrum"
        timeout = 300 if same_chain else 1500
        # Bridges can shave a little more than quoted; accept 75% of estimate as arrival.
        min_delta = max(funding.estimated_usdc * 0.75, HYPERLIQUID_BRIDGE_MIN_USDC)

        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            balance = await self._get_usdc_balance(funding.wallet_address)
            if balance is not None and balance - usdc_baseline >= min_delta:
                arrived = balance - usdc_baseline
                funding.arrived_usdc = arrived
                self._finish_step(step, f"{arrived:.2f} USDC arrived on Arbitrum")
                await self._persist(funding)
                return arrived
            await asyncio.sleep(10 if same_chain else 20)

        raise ValueError(
            f"Timed out waiting for USDC arrival on Arbitrum (expected ~{funding.estimated_usdc:.2f} USDC). "
            "The bridge transfer may still complete; check again before retrying."
        )

    async def _run_deposit_step(self, funding: FundingExecution, arrived_usdc: float) -> float:
        step = self._begin_step(funding, STEP_DEPOSIT, "Depositing USDC to Hyperliquid")
        await self._persist(funding)

        balance = await self._get_usdc_balance(funding.wallet_address)
        if balance is None:
            raise ValueError("Could not read Arbitrum USDC balance before deposit")

        deposit_amount = min(arrived_usdc, balance)
        min_deposit = max(HYPERLIQUID_BRIDGE_MIN_USDC, settings.YUKI_MIN_ALLOCATION_USDC)
        if deposit_amount < min_deposit:
            raise ValueError(
                f"Arrived USDC ({deposit_amount:.2f}) is below the {min_deposit:.2f} USDC Hyperliquid bridge minimum"
            )

        amount_units = int(deposit_amount * 1_000_000)
        bridge_address = settings.HYPERLIQUID_BRIDGE_ADDRESS
        calldata = (
            ERC20_TRANSFER_SELECTOR
            + bridge_address[2:].lower().rjust(64, "0")
            + hex(amount_units)[2:].rjust(64, "0")
        )

        tx_hash = await self._send_delegated_transaction(
            wallet_id=funding.wallet_id,
            chain_id=ARBITRUM_CHAIN_ID,
            transaction={"to": ARBITRUM_USDC_ADDRESS, "value": "0x0", "data": calldata},
        )
        step.tx_hash = tx_hash
        step.detail = f"Depositing {deposit_amount:.2f} USDC to the Hyperliquid bridge"
        await self._persist(funding)

        receipt_ok = await self._wait_for_receipt("arbitrum", tx_hash, timeout_seconds=300)
        if not receipt_ok:
            raise ValueError(f"Hyperliquid deposit transaction {tx_hash} failed or was not confirmed")

        funding.deposited_usdc = deposit_amount
        self._finish_step(step, f"{deposit_amount:.2f} USDC sent to the Hyperliquid bridge")
        await self._persist(funding)
        return deposit_amount

    async def _run_credit_step(self, funding: FundingExecution, hl_baseline: float, deposited_usdc: float):
        step = self._begin_step(funding, STEP_CREDIT, "Waiting for Hyperliquid to credit the deposit")
        await self._persist(funding)

        deadline = asyncio.get_event_loop().time() + 600
        min_delta = deposited_usdc * 0.95
        while asyncio.get_event_loop().time() < deadline:
            account_value = await self._get_hyperliquid_account_value(funding.wallet_address)
            if account_value is not None and account_value - hl_baseline >= min_delta:
                self._finish_step(step, f"Hyperliquid balance is now {account_value:.2f} USDC")
                await self._persist(funding)
                return
            await asyncio.sleep(15)

        raise ValueError(
            "Deposit confirmed on Arbitrum but Hyperliquid has not credited it yet. "
            "Credits normally land within minutes; check the Hyperliquid balance before retrying."
        )

    # ------------------------------------------------------------------
    # Privy delegated signing
    # ------------------------------------------------------------------

    async def _send_delegated_transaction(
        self,
        wallet_id: str,
        chain_id: int,
        transaction: Dict[str, Any],
    ) -> str:
        """Send a transaction from the user's embedded wallet via Privy session-signer RPC."""
        # Legacy delegated funding is retained for recovery only. Kata never
        # sponsors its gas; the active UI uses Circle Paymaster and charges the
        # user's USDC directly.
        return await self._privy_wallet_rpc(wallet_id, chain_id, transaction)

    async def _privy_wallet_rpc(
        self,
        wallet_id: str,
        chain_id: int,
        transaction: Dict[str, Any],
    ) -> str:
        privy_app_id = os.getenv("PRIVY_APP_ID") or settings.PRIVY_APP_ID
        privy_app_secret = os.getenv("PRIVY_APP_SECRET") or settings.PRIVY_APP_SECRET
        auth_key_id = os.getenv("PRIVY_AUTHORIZATION_KEY_ID")
        auth_key_private = os.getenv("PRIVY_AUTHORIZATION_KEY_PRIVATE_KEY")
        if not all([privy_app_id, privy_app_secret, auth_key_id, auth_key_private]):
            raise ValueError("Missing Privy delegation credentials (app id/secret, authorization key id/private key)")

        url = f"https://api.privy.io/v1/wallets/{wallet_id}/rpc"
        body: Dict[str, Any] = {
            "method": "eth_sendTransaction",
            "caip2": f"eip155:{chain_id}",
            "params": {"transaction": transaction},
        }
        headers = self._authorization_headers(
            url=url,
            body=body,
            privy_app_id=privy_app_id,
            privy_app_secret=privy_app_secret,
            auth_key_id=auth_key_id,
            auth_key_private=auth_key_private,
        )

        response = await self.http_client.post(url, headers=headers, json=body)
        if response.status_code != 200:
            raise ValueError(f"Privy RPC error {response.status_code}: {response.text}")

        result = response.json()
        tx_hash = result.get("result") or (result.get("data") or {}).get("hash")
        if not tx_hash:
            raise ValueError(f"Privy RPC returned no transaction hash: {result}")
        return tx_hash

    @staticmethod
    def _authorization_headers(
        url: str,
        body: Dict[str, Any],
        privy_app_id: str,
        privy_app_secret: str,
        auth_key_id: str,
        auth_key_private: str,
    ) -> Dict[str, str]:
        """Build Privy session-signer headers using Privy's SDK-compatible signer."""
        _ = auth_key_id
        signature = get_authorization_signature(
            url=url,
            body=body,
            method="POST",
            app_id=privy_app_id,
            private_key=auth_key_private,
        )
        basic_auth = base64.b64encode(f"{privy_app_id}:{privy_app_secret}".encode()).decode()

        return {
            "Authorization": f"Basic {basic_auth}",
            "privy-app-id": privy_app_id,
            "privy-authorization-signature": signature,
            "Content-Type": "application/json",
        }

    # ------------------------------------------------------------------
    # Chain reads
    # ------------------------------------------------------------------

    def _rpc_url(self, chain: str) -> str:
        config = CHAIN_CONFIG[chain]
        return os.getenv(config["rpc_env"]) or config["rpc_default"]

    async def _eth_rpc(self, chain: str, method: str, params: List[Any]) -> Any:
        response = await self.http_client.post(
            self._rpc_url(chain),
            json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("error"):
            raise ValueError(f"RPC error from {chain}: {payload['error']}")
        return payload.get("result")

    async def _get_native_balance_eth(self, chain: str, address: str) -> Optional[float]:
        try:
            result = await self._eth_rpc(chain, "eth_getBalance", [address, "latest"])
            return int(result, 16) / 10**18
        except Exception as e:
            logger.error(f"Failed to read {chain} ETH balance for {address}: {e}")
            return None

    async def _get_usdc_balance(self, address: str) -> Optional[float]:
        try:
            calldata = ERC20_BALANCE_OF_SELECTOR + address[2:].lower().rjust(64, "0")
            result = await self._eth_rpc(
                "arbitrum",
                "eth_call",
                [{"to": ARBITRUM_USDC_ADDRESS, "data": calldata}, "latest"],
            )
            return int(result, 16) / 1_000_000
        except Exception as e:
            logger.error(f"Failed to read Arbitrum USDC balance for {address}: {e}")
            return None

    async def _wait_for_receipt(self, chain: str, tx_hash: str, timeout_seconds: int) -> bool:
        deadline = asyncio.get_event_loop().time() + timeout_seconds
        while asyncio.get_event_loop().time() < deadline:
            try:
                receipt = await self._eth_rpc(chain, "eth_getTransactionReceipt", [tx_hash])
                if receipt:
                    return receipt.get("status") == "0x1"
            except Exception as e:
                logger.warning(f"Receipt poll failed for {tx_hash} on {chain}: {e}")
            await asyncio.sleep(5)
        return False

    async def _get_hyperliquid_account_value(self, address: str) -> Optional[float]:
        try:
            response = await self.http_client.post(
                "https://api.hyperliquid.xyz/info",
                json={"type": "clearinghouseState", "user": address},
            )
            response.raise_for_status()
            state = response.json() or {}
            summary = state.get("crossMarginSummary") or {}
            return float(summary.get("accountValue") or state.get("withdrawable") or 0.0)
        except Exception as e:
            logger.warning(f"Failed to read Hyperliquid account value for {address}: {e}")
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _min_gas_reserve(chain: str) -> float:
        # Mainnet gas is materially more expensive than the L2s.
        return 0.004 if chain == "ethereum" else 0.0002

    @staticmethod
    def _to_hex_quantity(value: Any) -> str:
        if isinstance(value, str) and value.startswith("0x"):
            return value
        return hex(int(str(value)))

    async def _get_active_delegation(self, user_id: str):
        try:
            from kata.services.delegation_service import get_delegation_service
            return await get_delegation_service().get_user_delegation(user_id)
        except Exception as e:
            logger.error(f"Could not load delegation for {user_id}: {e}")
            return None

    async def _lookup_embedded_wallet_id(self, user_id: str) -> Optional[str]:
        """Fallback wallet-id lookup for delegation records missing wallet_id."""
        try:
            privy_app_id = os.getenv("PRIVY_APP_ID") or settings.PRIVY_APP_ID
            privy_app_secret = os.getenv("PRIVY_APP_SECRET") or settings.PRIVY_APP_SECRET
            basic_auth = base64.b64encode(f"{privy_app_id}:{privy_app_secret}".encode()).decode()
            response = await self.http_client.get(
                f"https://api.privy.io/v1/users/{user_id}",
                headers={
                    "Authorization": f"Basic {basic_auth}",
                    "privy-app-id": privy_app_id,
                },
            )
            if response.status_code != 200:
                return None
            for account in response.json().get("linked_accounts", []):
                if account.get("type") == "wallet" and account.get("imported") is False:
                    return account.get("id")
            return None
        except Exception as e:
            logger.error(f"Failed to look up embedded wallet id for {user_id}: {e}")
            return None

    def _begin_step(self, funding: FundingExecution, name: str, detail: str) -> FundingStep:
        funding.current_step = name
        step = funding.step(name)
        step.status = "running"
        step.detail = detail
        step.started_at = datetime.now(timezone.utc).isoformat()
        funding.updated_at = datetime.now(timezone.utc).isoformat()
        return step

    @staticmethod
    def _finish_step(step: FundingStep, detail: str):
        step.status = "completed"
        step.detail = detail
        step.finished_at = datetime.now(timezone.utc).isoformat()

    async def _persist(self, funding: FundingExecution):
        funding.updated_at = datetime.now(timezone.utc).isoformat()
        if not self.supabase:
            return
        try:
            record = funding.to_dict()
            await asyncio.to_thread(
                lambda: self.supabase.table("yuki_funding_executions")
                .upsert(record, on_conflict="funding_id")
                .execute()
            )
        except Exception as e:
            logger.warning(f"Could not persist funding execution {funding.funding_id}: {e}")

    def _load_persisted(self, funding_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        if not self.supabase:
            return None
        try:
            result = (
                self.supabase.table("yuki_funding_executions")
                .select("*")
                .eq("funding_id", funding_id)
                .eq("user_id", user_id)
                .execute()
            )
            if result.data:
                record = result.data[0]
                if isinstance(record.get("steps"), str):
                    record["steps"] = json.loads(record["steps"])
                return record
            return None
        except Exception as e:
            logger.warning(f"Could not load funding execution {funding_id}: {e}")
            return None


# Global service instance
_yuki_funding_service: Optional[YukiFundingService] = None


def get_yuki_funding_service() -> YukiFundingService:
    """Get or create the global Yuki funding service."""
    global _yuki_funding_service
    if _yuki_funding_service is None:
        _yuki_funding_service = YukiFundingService()
    return _yuki_funding_service
