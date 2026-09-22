"""
Onchain Agent Service for Flow AI Trading Platform.

Manages the integration between smart contracts and AI agents,
handling delegation, execution, and monitoring of onchain agent activities.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal
from dataclasses import dataclass
from datetime import datetime

from web3 import Web3
from web3.contract import Contract
from eth_account import Account
from eth_utils import to_checksum_address

from kata.config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class AgentDelegation:
    """Represents a user's delegation to an agent"""
    user_address: str
    agent_type: str  # 'sakura', 'ryu', 'yuki'
    amount_usd: float
    transaction_hash: str
    block_number: int
    timestamp: datetime
    is_active: bool

@dataclass
class AgentExecution:
    """Represents an agent execution"""
    execution_id: int
    user_address: str
    agent_type: str
    action_hash: str
    amount: float
    timestamp: datetime
    success: bool

@dataclass
class ContractAddresses:
    """Contract addresses on Base"""
    flow_agent_vault: str
    sakura_pendle_vault: str
    hyperliquid_vault: str
    agent_executor: str

# Base mainnet contract addresses (to be updated with deployed addresses)
BASE_CONTRACTS = ContractAddresses(
    flow_agent_vault="0x0000000000000000000000000000000000000000",  # Update after deployment
    sakura_pendle_vault="0x0000000000000000000000000000000000000000",  # Update after deployment
    hyperliquid_vault="0x0000000000000000000000000000000000000000",  # Update after deployment
    agent_executor="0x0000000000000000000000000000000000000000"  # Update after deployment
)

class OnchainAgentService:
    """Service for managing onchain agent interactions"""

    def __init__(self):
        """Initialize the onchain agent service"""
        self.w3 = self._setup_web3()
        self.contracts = self._setup_contracts()
        self.account = self._setup_account()

        # Agent type mappings
        self.agent_type_mapping = {
            'sakura': 0,
            'ryu': 1,
            'yuki': 2
        }

        logger.info("OnchainAgentService initialized")

    def _setup_web3(self) -> Web3:
        """Setup Web3 connection to Base network"""
        rpc_url = f"https://base-mainnet.g.alchemy.com/v2/{settings.VITE_ALCHEMY_API_KEY}"
        w3 = Web3(Web3.HTTPProvider(rpc_url))

        if not w3.is_connected():
            raise ConnectionError("Failed to connect to Base network")

        logger.info(f"Connected to Base network, latest block: {w3.eth.block_number}")
        return w3

    def _setup_contracts(self) -> Dict[str, Contract]:
        """Setup contract instances"""
        contracts = {}

        # Load contract ABIs (these would be generated from compiled contracts)
        # For now, using simplified ABIs - in production, load from artifacts

        flow_vault_abi = [
            {
                "name": "depositAndDelegate",
                "type": "function",
                "inputs": [
                    {"name": "agentType", "type": "uint8"},
                    {"name": "delegateAmountUSD", "type": "uint256"},
                    {"name": "minUSDCOut", "type": "uint256"}
                ],
                "outputs": [],
                "stateMutability": "payable"
            },
            {
                "name": "FundsDelegated",
                "type": "event",
                "inputs": [
                    {"name": "user", "type": "address", "indexed": True},
                    {"name": "agentType", "type": "uint8", "indexed": True},
                    {"name": "amount", "type": "uint256"}
                ]
            }
        ]

        agent_executor_abi = [
            {
                "name": "executeSakuraStrategy",
                "type": "function",
                "inputs": [
                    {"name": "user", "type": "address"},
                    {"name": "pendleMarket", "type": "address"},
                    {"name": "amount", "type": "uint256"}
                ],
                "outputs": [],
                "stateMutability": "nonpayable"
            },
            {
                "name": "executeRyuTrade",
                "type": "function",
                "inputs": [
                    {"name": "user", "type": "address"},
                    {"name": "symbol", "type": "string"},
                    {"name": "tradeType", "type": "uint8"},
                    {"name": "amount", "type": "uint256"},
                    {"name": "targetPrice", "type": "uint256"}
                ],
                "outputs": [],
                "stateMutability": "nonpayable"
            },
            {
                "name": "AgentExecutionCompleted",
                "type": "event",
                "inputs": [
                    {"name": "user", "type": "address", "indexed": True},
                    {"name": "agentType", "type": "uint8", "indexed": True},
                    {"name": "executionId", "type": "uint256"},
                    {"name": "success", "type": "bool"}
                ]
            }
        ]

        try:
            contracts['flow_vault'] = self.w3.eth.contract(
                address=to_checksum_address(BASE_CONTRACTS.flow_agent_vault),
                abi=flow_vault_abi
            )

            contracts['agent_executor'] = self.w3.eth.contract(
                address=to_checksum_address(BASE_CONTRACTS.agent_executor),
                abi=agent_executor_abi
            )

            logger.info("Contract instances created successfully")

        except Exception as e:
            logger.error(f"Failed to setup contracts: {e}")
            raise

        return contracts

    def _setup_account(self) -> Account:
        """Setup backend account for contract interactions"""
        # In production, load from secure environment variable
        private_key = settings.FLOW_BACKEND_PRIVATE_KEY  # Set in environment
        account = Account.from_key(private_key)

        logger.info(f"Backend account loaded: {account.address}")
        return account

    # Agent Delegation Methods
    async def get_user_delegations(self, user_address: str) -> List[AgentDelegation]:
        """Get all delegations for a user"""
        try:
            # Query FundsDelegated events for user
            contract = self.contracts['flow_vault']

            # Get events from recent blocks (can be optimized with indexing)
            from_block = self.w3.eth.block_number - 10000  # Last ~10k blocks

            event_filter = contract.events.FundsDelegated.create_filter(
                fromBlock=from_block,
                argument_filters={'user': to_checksum_address(user_address)}
            )

            events = event_filter.get_all_entries()

            delegations = []
            for event in events:
                agent_type_num = event.args.agentType
                agent_type = self._get_agent_type_string(agent_type_num)

                delegation = AgentDelegation(
                    user_address=user_address,
                    agent_type=agent_type,
                    amount_usd=float(event.args.amount / 1e6),  # Convert from USDC decimals
                    transaction_hash=event.transactionHash.hex(),
                    block_number=event.blockNumber,
                    timestamp=datetime.fromtimestamp(
                        self.w3.eth.get_block(event.blockNumber).timestamp
                    ),
                    is_active=True  # Would need additional logic to determine
                )
                delegations.append(delegation)

            logger.info(f"Found {len(delegations)} delegations for user {user_address}")
            return delegations

        except Exception as e:
            logger.error(f"Error getting user delegations: {e}")
            return []

    # Agent Execution Methods
    async def execute_sakura_strategy(
        self,
        user_address: str,
        pendle_market: str,
        amount_usd: float
    ) -> Optional[str]:
        """Execute Sakura Pendle yield strategy"""
        try:
            contract = self.contracts['agent_executor']
            amount_wei = int(amount_usd * 1e6)  # Convert to USDC decimals

            # Build transaction
            txn = contract.functions.executeSakuraStrategy(
                to_checksum_address(user_address),
                to_checksum_address(pendle_market),
                amount_wei
            ).build_transaction({
                'from': self.account.address,
                'gas': 500000,  # Estimate gas in production
                'gasPrice': self.w3.eth.gas_price,
                'nonce': self.w3.eth.get_transaction_count(self.account.address)
            })

            # Sign and send transaction
            signed_txn = self.account.sign_transaction(txn)
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)

            logger.info(f"Sakura strategy execution sent: {tx_hash.hex()}")

            # Wait for confirmation (optional - can be done async)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

            if receipt.status == 1:
                logger.info(f"Sakura strategy executed successfully: {tx_hash.hex()}")
                return tx_hash.hex()
            else:
                logger.error(f"Sakura strategy execution failed: {tx_hash.hex()}")
                return None

        except Exception as e:
            logger.error(f"Error executing Sakura strategy: {e}")
            return None

    async def execute_ryu_trade(
        self,
        user_address: str,
        symbol: str,
        is_buy: bool,
        amount_usd: float,
        target_price: float
    ) -> Optional[str]:
        """Execute Ryu spot trade"""
        try:
            contract = self.contracts['agent_executor']

            trade_type = 0 if is_buy else 1  # 0 = SPOT_BUY, 1 = SPOT_SELL
            amount_wei = int(amount_usd * 1e6)
            target_price_wei = int(target_price * 1e6)

            txn = contract.functions.executeRyuTrade(
                to_checksum_address(user_address),
                symbol,
                trade_type,
                amount_wei,
                target_price_wei
            ).build_transaction({
                'from': self.account.address,
                'gas': 300000,
                'gasPrice': self.w3.eth.gas_price,
                'nonce': self.w3.eth.get_transaction_count(self.account.address)
            })

            signed_txn = self.account.sign_transaction(txn)
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)

            logger.info(f"Ryu trade execution sent: {tx_hash.hex()}")

            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

            if receipt.status == 1:
                logger.info(f"Ryu trade executed successfully: {tx_hash.hex()}")
                return tx_hash.hex()
            else:
                logger.error(f"Ryu trade execution failed: {tx_hash.hex()}")
                return None

        except Exception as e:
            logger.error(f"Error executing Ryu trade: {e}")
            return None

    # Event Monitoring
    async def start_event_monitoring(self):
        """Start monitoring contract events"""
        logger.info("Starting contract event monitoring...")

        # Create event filters
        delegation_filter = self.contracts['flow_vault'].events.FundsDelegated.create_filter(
            fromBlock='latest'
        )

        execution_filter = self.contracts['agent_executor'].events.AgentExecutionCompleted.create_filter(
            fromBlock='latest'
        )

        # Monitor events in background
        asyncio.create_task(self._monitor_events(delegation_filter, execution_filter))

    async def _monitor_events(self, delegation_filter, execution_filter):
        """Monitor contract events continuously"""
        while True:
            try:
                # Check for new delegation events
                for event in delegation_filter.get_new_entries():
                    await self._handle_delegation_event(event)

                # Check for new execution events
                for event in execution_filter.get_new_entries():
                    await self._handle_execution_event(event)

                await asyncio.sleep(5)  # Poll every 5 seconds

            except Exception as e:
                logger.error(f"Error monitoring events: {e}")
                await asyncio.sleep(10)  # Wait longer on error

    async def _handle_delegation_event(self, event):
        """Handle new delegation event"""
        user = event.args.user
        agent_type_num = event.args.agentType
        amount = event.args.amount

        agent_type = self._get_agent_type_string(agent_type_num)
        amount_usd = float(amount / 1e6)

        logger.info(f"New delegation: {user} -> {agent_type}: ${amount_usd}")

        # For Hyperliquid agents, create HL account
        if agent_type in ['ryu', 'yuki']:
            await self._create_hyperliquid_account(user, agent_type, amount_usd)

    async def _handle_execution_event(self, event):
        """Handle execution completion event"""
        user = event.args.user
        agent_type_num = event.args.agentType
        execution_id = event.args.executionId
        success = event.args.success

        agent_type = self._get_agent_type_string(agent_type_num)

        logger.info(f"Execution completed: {user} {agent_type} #{execution_id} - {'Success' if success else 'Failed'}")

    async def _create_hyperliquid_account(self, user_address: str, agent_type: str, amount: float):
        """Create Hyperliquid account for user (placeholder for HL integration)"""
        logger.info(f"Creating HL account for {user_address} {agent_type} with ${amount}")

        # This would integrate with actual Hyperliquid account creation
        # For now, just emit the contract event to signal backend
        try:
            hl_contract = self.contracts.get('hyperliquid_vault')
            if hl_contract:
                # Call createHLAccount function
                pass
        except Exception as e:
            logger.error(f"Error creating HL account: {e}")

    # Utility Methods
    def _get_agent_type_string(self, agent_type_num: int) -> str:
        """Convert agent type number to string"""
        reverse_mapping = {v: k for k, v in self.agent_type_mapping.items()}
        return reverse_mapping.get(agent_type_num, 'unknown')

    async def get_contract_balance(self, contract_address: str) -> float:
        """Get USDC balance of a contract"""
        try:
            # USDC contract on Base
            usdc_abi = [
                {
                    "name": "balanceOf",
                    "type": "function",
                    "inputs": [{"name": "account", "type": "address"}],
                    "outputs": [{"name": "", "type": "uint256"}],
                    "stateMutability": "view"
                }
            ]

            usdc_contract = self.w3.eth.contract(
                address=to_checksum_address("0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"),
                abi=usdc_abi
            )

            balance_wei = usdc_contract.functions.balanceOf(
                to_checksum_address(contract_address)
            ).call()

            return float(balance_wei / 1e6)  # Convert from USDC decimals

        except Exception as e:
            logger.error(f"Error getting contract balance: {e}")
            return 0.0

# Global instance
onchain_agent_service = OnchainAgentService()