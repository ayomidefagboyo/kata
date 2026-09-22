"""
LiFi Bridge Service for Real-Time Cross-Chain Bridge Quotes

This service integrates with LiFi API to provide:
- Real-time bridge quotes from supported source chains to Arbitrum
- Live pricing with accurate fees and timing
- Multiple bridge provider options (deBridge, Stargate, etc.)
"""

import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass
import httpx

logger = logging.getLogger(__name__)

SOLANA_CHAIN_ID = 1151111081099710
SOLANA_USDC_ADDRESS = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"

@dataclass
class LiFiBridgeQuote:
    """LiFi bridge quote data structure."""
    input_amount: float
    output_amount: float
    total_fees: float
    bridge_fee: float
    gas_fee: float
    estimated_time: int  # seconds
    tool_name: str  # Bridge provider (deBridge, Stargate, etc.)
    slippage: float
    quote_id: str
    route_data: Dict[str, Any]
    valid_until: datetime


@dataclass
class LiFiSwapQuote:
    """LiFi swap quote with everything needed to actually execute the trade.

    Unlike LiFiBridgeQuote (built for display estimates and tolerant of a
    synthesized fallback), this carries the real ready-to-sign transaction.
    get_swap_quote() never fabricates one of these on error -- it returns
    None so callers abort rather than sign something that isn't real.
    """
    from_chain_id: int
    to_chain_id: int
    from_token_address: str
    to_token_address: str
    from_amount_base_units: str  # raw integer string in the token's base units
    from_amount_decimal: float
    to_amount_decimal: float
    to_token_decimals: int
    min_to_amount_decimal: float  # to_amount after the requested slippage tolerance
    approval_address: Optional[str]  # spender to approve for from_token; None if native asset
    transaction_request: Dict[str, Any]  # {to, data, value, chainId, gasLimit, ...} - ready to sign
    tool_name: str
    quote_id: str
    price_impact: Optional[float]
    valid_until: datetime
    transaction_kind: str = "evm"
    from_address: str = ""
    to_address: str = ""
    from_amount_for_gas: Optional[str] = None


class LiFiBridgeService:
    """
    LiFi integration service for real-time bridge quotes and transactions.

    Provides accurate pricing for ETH -> USDC routing into Arbitrum USDC.
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize LiFi service."""
        self.api_key = api_key
        self.base_url = "https://li.quest/v1"
        headers = {
            "accept": "application/json",
            "content-type": "application/json"
        }
        if api_key:
            headers["x-lifi-api-key"] = api_key
        self.http_client = httpx.AsyncClient(
            timeout=30.0,
            headers=headers,
        )

        # Floww's first safe universal funding routes. Keep this explicit so
        # unsupported source assets are not misquoted as native ETH.
        self.chains = {
            "ethereum": {
                "id": 1,
                "name": "Ethereum",
                "nativeCurrency": "ETH",
                "tokens": {
                    "ETH": "0x0000000000000000000000000000000000000000",
                    "USDC": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
                }
            },
            "base": {
                "id": 8453,
                "name": "Base",
                "nativeCurrency": "ETH",
                "tokens": {
                    "ETH": "0x0000000000000000000000000000000000000000",
                    "USDC": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
                }
            },
            "optimism": {
                "id": 10,
                "name": "Optimism",
                "nativeCurrency": "ETH",
                "tokens": {
                    "ETH": "0x0000000000000000000000000000000000000000",
                    "USDC": "0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85"
                }
            },
            "arbitrum": {
                "id": 42161,
                "name": "Arbitrum",
                "nativeCurrency": "ETH",
                "tokens": {
                    "ETH": "0x0000000000000000000000000000000000000000",
                    "USDC": "0xaf88d065e77c8cc2239327c5edb3a432268e5831"
                }
            }
        }
        self.chain_aliases = {
            "mainnet": "ethereum",
            "eth": "ethereum",
            "ethereum-mainnet": "ethereum",
            "arbitrum-one": "arbitrum",
            "arb": "arbitrum",
            "op": "optimism",
            "optimistic-ethereum": "optimism",
        }

        logger.info("LiFi bridge service initialized")

    def normalize_chain(self, chain: str) -> str:
        """Normalize user/API chain names to configured LiFi chain slugs."""
        chain_key = chain.lower().strip().replace(" ", "-")
        return self.chain_aliases.get(chain_key, chain_key)

    def get_configured_chains(self) -> List[Dict[str, Any]]:
        """Return source chains Floww can safely quote for ETH funding today."""
        return [
            {
                "id": config["id"],
                "slug": slug,
                "name": config["name"],
                "nativeCurrency": config["nativeCurrency"],
                "supports_eth_funding": config["nativeCurrency"] == "ETH" and "ETH" in config["tokens"],
            }
            for slug, config in self.chains.items()
        ]

    async def get_bridge_quote(
        self,
        from_chain: str,
        to_chain: str,
        from_token: str,
        to_token: str,
        amount: float,
        wallet_address: str
    ) -> Optional[LiFiBridgeQuote]:
        """
        Get real-time bridge quote from LiFi.

        Args:
            from_chain: Source chain (e.g., "base")
            to_chain: Destination chain (e.g., "arbitrum")
            from_token: Source token (e.g., "ETH")
            to_token: Destination token (e.g., "USDC")
            amount: Amount to bridge
            wallet_address: User's wallet address

        Returns:
            LiFiBridgeQuote with real pricing data
        """
        try:
            from_chain = self.normalize_chain(from_chain)
            to_chain = self.normalize_chain(to_chain)
            from_token = from_token.upper()
            to_token = to_token.upper()
            from_chain_config = self.chains.get(from_chain)
            to_chain_config = self.chains.get(to_chain)

            if not from_chain_config or not to_chain_config:
                logger.error(f"Unsupported chain: {from_chain} -> {to_chain}")
                return None

            from_token_address = from_chain_config["tokens"].get(from_token)
            to_token_address = to_chain_config["tokens"].get(to_token)

            if not from_token_address or not to_token_address:
                logger.error(f"Unsupported token: {from_token} -> {to_token}")
                return None

            # Convert amount to wei based on token decimals
            if from_token == "ETH":
                amount_wei = str(int(amount * 10**18))
            else:  # USDC
                amount_wei = str(int(amount * 10**6))

            # LiFi quote request
            quote_params = {
                "fromChain": from_chain_config["id"],
                "toChain": to_chain_config["id"],
                "fromToken": from_token_address,
                "toToken": to_token_address,
                "fromAmount": amount_wei,
                "fromAddress": wallet_address,
                "toAddress": wallet_address,
                "options": {
                    "slippage": 0.03,  # 3% slippage tolerance
                    "allowBridges": ["deBridge", "stargate", "hyphen", "across"],
                    "allowExchanges": ["1inch", "0x"],
                    "preferBridges": ["deBridge", "stargate"],  # Prefer reliable bridges
                }
            }

            logger.info(f"Getting LiFi quote: {amount} {from_token} on {from_chain} -> {to_token} on {to_chain}")

            # Call LiFi quote API
            response = await self.http_client.get(
                f"{self.base_url}/quote",
                params=quote_params
            )

            if response.status_code != 200:
                logger.error(f"LiFi API error: {response.status_code} - {response.text}")
                return await self._get_fallback_quote(from_chain, to_chain, from_token, to_token, amount)

            quote_data = response.json()

            # Parse LiFi response
            estimate = quote_data.get("estimate", {})
            action = quote_data.get("action", {})
            tool_details = quote_data.get("toolDetails", {})

            # Extract amounts (convert from wei)
            from_amount_wei = int(estimate.get("fromAmount", 0))
            to_amount_wei = int(estimate.get("toAmount", 0))

            if from_token == "ETH":
                input_amount = from_amount_wei / 10**18
            else:
                input_amount = from_amount_wei / 10**6

            if to_token == "USDC":
                output_amount = to_amount_wei / 10**6
            else:
                output_amount = to_amount_wei / 10**18

            # Extract fees
            fees_breakdown = estimate.get("feeCosts", [])
            gas_costs = estimate.get("gasCosts", [])

            # Calculate total fees
            bridge_fee = 0
            gas_fee = 0

            for fee in fees_breakdown:
                if fee.get("included", True):
                    if fee.get("token") == from_token_address:
                        if from_token == "ETH":
                            bridge_fee += int(fee.get("amount", 0)) / 10**18
                        else:
                            bridge_fee += int(fee.get("amount", 0)) / 10**6

            for gas_cost in gas_costs:
                if gas_cost.get("type") == "SEND":
                    gas_fee += int(gas_cost.get("amount", 0)) / 10**18  # Gas always in ETH

            total_fees = bridge_fee + gas_fee

            # Extract timing and tool info
            estimated_time = estimate.get("executionDuration", 180)  # Default 3 minutes
            tool_name = tool_details.get("name", "Unknown")

            # Create quote object
            quote = LiFiBridgeQuote(
                input_amount=input_amount,
                output_amount=output_amount,
                total_fees=total_fees,
                bridge_fee=bridge_fee,
                gas_fee=gas_fee,
                estimated_time=estimated_time,
                tool_name=tool_name,
                slippage=quote_params["options"]["slippage"],
                quote_id=quote_data.get("id", f"quote_{int(time.time())}"),
                route_data=quote_data,
                valid_until=datetime.now()
            )

            logger.info(
                f"LiFi quote: {input_amount:.6f} {from_token} -> {output_amount:.6f} {to_token} "
                f"(fees: {total_fees:.6f}, time: {estimated_time}s, tool: {tool_name})"
            )

            return quote

        except Exception as e:
            logger.error(f"Error getting LiFi quote: {e}")
            from_chain = self.normalize_chain(from_chain)
            to_chain = self.normalize_chain(to_chain)
            from_token = from_token.upper()
            to_token = to_token.upper()
            return await self._get_fallback_quote(from_chain, to_chain, from_token, to_token, amount)

    async def _get_fallback_quote(
        self,
        from_chain: str,
        to_chain: str,
        from_token: str,
        to_token: str,
        amount: float
    ) -> LiFiBridgeQuote:
        """Get fallback quote when LiFi API is unavailable."""
        try:
            # Estimate conversion for ETH -> USDC
            if from_token == "ETH" and to_token == "USDC":
                # Fetch current ETH price (fallback to $3000)
                eth_price = await self._get_eth_price() or 3000.0
                usdc_output = amount * eth_price

                is_same_chain_swap = from_chain == to_chain

                # Estimate fees. Same-chain Arbitrum swaps are much cheaper than bridges.
                bridge_fee_usd = usdc_output * (0.003 if is_same_chain_swap else 0.005)
                gas_fee_eth = 0.0003 if is_same_chain_swap and from_chain == "arbitrum" else 0.002
                gas_fee_usd = gas_fee_eth * eth_price

                total_fees_usd = bridge_fee_usd + gas_fee_usd
                net_usdc = usdc_output - total_fees_usd

                return LiFiBridgeQuote(
                    input_amount=amount,
                    output_amount=max(0, net_usdc),
                    total_fees=total_fees_usd,
                    bridge_fee=bridge_fee_usd,
                    gas_fee=gas_fee_usd,
                    estimated_time=45 if is_same_chain_swap else 180,
                    tool_name="Estimated (Arbitrum swap)" if is_same_chain_swap else "Estimated (deBridge)",
                    slippage=0.03,
                    quote_id=f"fallback_{int(time.time())}",
                    route_data={"fallback": True},
                    valid_until=datetime.now()
                )
            else:
                # Generic fallback
                return LiFiBridgeQuote(
                    input_amount=amount,
                    output_amount=amount * 0.995,  # 0.5% fee estimate
                    total_fees=amount * 0.005,
                    bridge_fee=amount * 0.003,
                    gas_fee=amount * 0.002,
                    estimated_time=180,
                    tool_name="Estimated",
                    slippage=0.03,
                    quote_id=f"fallback_{int(time.time())}",
                    route_data={"fallback": True},
                    valid_until=datetime.now()
                )

        except Exception as e:
            logger.error(f"Error creating fallback quote: {e}")
            # Return minimal fallback
            return LiFiBridgeQuote(
                input_amount=amount,
                output_amount=0,
                total_fees=0,
                bridge_fee=0,
                gas_fee=0,
                estimated_time=180,
                tool_name="Error",
                slippage=0.03,
                quote_id=f"error_{int(time.time())}",
                route_data={"error": str(e)},
                valid_until=datetime.now()
            )

    async def _get_eth_price(self) -> Optional[float]:
        """Get current ETH price from a simple API."""
        try:
            response = await self.http_client.get(
                "https://api.coingecko.com/api/v3/simple/price",
                params={"ids": "ethereum", "vs_currencies": "usd"},
                timeout=10.0
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("ethereum", {}).get("usd")

        except Exception as e:
            logger.warning(f"Failed to fetch ETH price: {e}")

        return None

    async def get_supported_chains(self) -> List[Dict[str, Any]]:
        """Get list of supported chains from LiFi."""
        try:
            response = await self.http_client.get(f"{self.base_url}/chains")

            if response.status_code == 200:
                chains_data = response.json()
                return [
                    {
                        "id": chain.get("id"),
                        "name": chain.get("name"),
                        "logo": chain.get("logoURI"),
                        "nativeCurrency": chain.get("nativeCurrency", {}).get("symbol")
                    }
                    for chain in chains_data.get("chains", [])
                ]

        except Exception as e:
            logger.error(f"Error fetching supported chains: {e}")

        return []

    async def get_supported_tools(self) -> List[Dict[str, Any]]:
        """Get list of supported bridge tools from LiFi."""
        try:
            response = await self.http_client.get(f"{self.base_url}/tools")

            if response.status_code == 200:
                tools_data = response.json()
                return [
                    {
                        "name": tool.get("name"),
                        "logo": tool.get("logoURI"),
                        "description": tool.get("description")
                    }
                    for tool in tools_data.get("tools", [])
                ]

        except Exception as e:
            logger.error(f"Error fetching supported tools: {e}")

        return []

    async def get_swap_quote(
        self,
        chain_id: int,
        from_token_address: str,
        to_token_address: str,
        from_amount_base_units: str,
        wallet_address: str,
        from_token_decimals: int = 6,
        to_token_decimals: Optional[int] = None,
        slippage: float = 0.03,
    ) -> Optional[LiFiSwapQuote]:
        """
        Get a same-chain swap quote from LiFi, including the ready-to-sign
        transaction request and (if the source token needs it) the approval
        spender address.

        Unlike get_bridge_quote, this NEVER falls back to a synthesized
        estimate on error -- a fallback quote has no real transaction to
        sign, so any caller that executed one would sign nothing or (worse)
        something fabricated. On any failure this returns None; the caller
        must abort the trade rather than guess.

        Args:
            chain_id: EVM chain ID for both sides of the swap (same-chain only for now).
            from_token_address: Source token contract address (checksummed or lowercase).
            to_token_address: Destination token contract address.
            from_amount_base_units: Amount to swap, as an integer string in the
                source token's base units (e.g. USDC's 6-decimal units).
            wallet_address: The wallet that will sign and hold both sides of the swap.
            from_token_decimals: Decimals of the source token, for display math.
            to_token_decimals: Decimals of the destination token, if known ahead of
                time. If omitted, best-effort parsed from the quote response.
            slippage: Max acceptable slippage fraction (0.03 = 3%).

        Returns:
            LiFiSwapQuote with the transaction_request ready to sign, or None
            if no real quote could be obtained.
        """
        try:
            quote_params = {
                "fromChain": chain_id,
                "toChain": chain_id,
                "fromToken": from_token_address,
                "toToken": to_token_address,
                "fromAmount": from_amount_base_units,
                "fromAddress": wallet_address,
                "toAddress": wallet_address,
                "slippage": slippage,
            }

            logger.info(
                "Getting LiFi swap quote: %s base units of %s -> %s on chain %s",
                from_amount_base_units, from_token_address, to_token_address, chain_id,
            )

            response = await self.http_client.get(f"{self.base_url}/quote", params=quote_params)

            if response.status_code != 200:
                logger.error(f"LiFi swap quote API error: {response.status_code} - {response.text}")
                return None

            quote_data = response.json()

            transaction_request = quote_data.get("transactionRequest")
            if not transaction_request or not transaction_request.get("to") or not transaction_request.get("data"):
                logger.error("LiFi swap quote response missing a usable transactionRequest")
                return None

            estimate = quote_data.get("estimate", {}) or {}
            tool_details = quote_data.get("toolDetails", {}) or {}
            action = quote_data.get("action", {}) or {}

            resolved_to_decimals = to_token_decimals
            if resolved_to_decimals is None:
                to_token_meta = (action.get("toToken") or {})
                resolved_to_decimals = int(to_token_meta.get("decimals") or 18)

            from_amount_units = int(estimate.get("fromAmount") or from_amount_base_units)
            to_amount_units = int(estimate.get("toAmount") or 0)
            to_amount_min_units = int(estimate.get("toAmountMin") or to_amount_units)

            from_amount_decimal = from_amount_units / (10 ** from_token_decimals)
            to_amount_decimal = to_amount_units / (10 ** resolved_to_decimals)
            min_to_amount_decimal = to_amount_min_units / (10 ** resolved_to_decimals)

            approval_address = estimate.get("approvalAddress")

            price_impact_raw = estimate.get("priceImpact") if isinstance(estimate, dict) else None
            try:
                price_impact = float(price_impact_raw) if price_impact_raw is not None else None
            except (TypeError, ValueError):
                price_impact = None

            quote = LiFiSwapQuote(
                from_chain_id=chain_id,
                to_chain_id=chain_id,
                from_token_address=from_token_address,
                to_token_address=to_token_address,
                from_amount_base_units=str(from_amount_units),
                from_amount_decimal=from_amount_decimal,
                to_amount_decimal=to_amount_decimal,
                to_token_decimals=resolved_to_decimals,
                min_to_amount_decimal=min_to_amount_decimal,
                approval_address=approval_address,
                transaction_request=transaction_request,
                tool_name=tool_details.get("name", "Unknown"),
                quote_id=quote_data.get("id", f"swap_{int(time.time())}"),
                price_impact=price_impact,
                valid_until=datetime.now(),
            )

            logger.info(
                "LiFi swap quote: %.6f base units -> %.6f (min %.6f) via %s",
                from_amount_decimal, to_amount_decimal, min_to_amount_decimal, quote.tool_name,
            )

            return quote

        except Exception as e:
            logger.error(f"Error getting LiFi swap quote: {e}")
            return None

    async def get_token_transfer_quote(
        self,
        from_chain_id: int,
        to_chain_id: int,
        from_token_address: str,
        to_token_address: str,
        from_amount_base_units: str,
        from_address: str,
        to_address: str,
        from_token_decimals: int,
        to_token_decimals: Optional[int] = None,
        slippage: float = 0.03,
        from_amount_for_gas: Optional[str] = None,
        max_price_impact: Optional[float] = None,
    ) -> Optional[LiFiSwapQuote]:
        """Return one strict, executable any-to-any LI.FI quote.

        EVM-source quotes carry ``to``/``data`` transaction fields. A
        Solana-source quote carries only base64 serialized transaction data.
        This method accepts both shapes but never fabricates a fallback.
        """
        try:
            amount_int = int(from_amount_base_units)
            if amount_int <= 0:
                raise ValueError("from_amount_base_units must be positive")
            gas_amount_int = int(from_amount_for_gas or "0")
            if gas_amount_int < 0 or gas_amount_int >= amount_int:
                raise ValueError("from_amount_for_gas must be smaller than fromAmount")

            quote_params: Dict[str, Any] = {
                "fromChain": from_chain_id,
                "toChain": to_chain_id,
                "fromToken": from_token_address,
                "toToken": to_token_address,
                "fromAmount": str(amount_int),
                "fromAddress": from_address,
                "toAddress": to_address,
                "slippage": slippage,
                "order": "RECOMMENDED",
            }
            if gas_amount_int:
                quote_params["fromAmountForGas"] = str(gas_amount_int)
            if max_price_impact is not None:
                quote_params["maxPriceImpact"] = max_price_impact

            response = await self.http_client.get(
                f"{self.base_url}/quote",
                params=quote_params,
            )
            if response.status_code != 200:
                logger.warning(
                    "LI.FI token-transfer quote unavailable (%s): %s",
                    response.status_code,
                    response.text,
                )
                return None

            quote_data = response.json()
            action = quote_data.get("action") or {}
            estimate = quote_data.get("estimate") or {}
            transaction_request = quote_data.get("transactionRequest") or {}
            transaction_data = transaction_request.get("data")
            transaction_kind = (
                "solana" if int(from_chain_id) == SOLANA_CHAIN_ID else "evm"
            )
            if not transaction_data:
                logger.error("LI.FI quote is missing executable transaction data")
                return None
            if transaction_kind == "evm" and not transaction_request.get("to"):
                logger.error("LI.FI EVM quote is missing transaction target")
                return None

            action_from_chain = int(action.get("fromChainId") or 0)
            action_to_chain = int(action.get("toChainId") or 0)
            action_from_amount = str(
                action.get("fromAmount")
                or estimate.get("fromAmount")
                or ""
            )
            if (
                action_from_chain != int(from_chain_id)
                or action_to_chain != int(to_chain_id)
                or action_from_amount != str(amount_int)
            ):
                logger.error("LI.FI quote action does not match the requested route")
                return None

            action_from_address = str(action.get("fromAddress") or "")
            action_to_address = str(action.get("toAddress") or "")
            from_address_matches = (
                action_from_address == from_address
                if from_chain_id == SOLANA_CHAIN_ID
                else action_from_address.lower() == from_address.lower()
            )
            to_address_matches = (
                action_to_address == to_address
                if to_chain_id == SOLANA_CHAIN_ID
                else action_to_address.lower() == to_address.lower()
            )
            transaction_chain = int(
                transaction_request.get("chainId") or from_chain_id
            )
            transaction_from = str(
                transaction_request.get("from") or from_address
            )
            transaction_from_matches = (
                transaction_from == from_address
                if from_chain_id == SOLANA_CHAIN_ID
                else transaction_from.lower() == from_address.lower()
            )
            if (
                not from_address_matches
                or not to_address_matches
                or transaction_chain != int(from_chain_id)
                or not transaction_from_matches
            ):
                logger.error("LI.FI quote addresses do not match the requested route")
                return None

            action_from_token = str((action.get("fromToken") or {}).get("address") or "")
            action_to_token = str((action.get("toToken") or {}).get("address") or "")
            from_token_matches = (
                action_from_token == from_token_address
                if from_chain_id == SOLANA_CHAIN_ID
                else action_from_token.lower() == from_token_address.lower()
            )
            to_token_matches = (
                action_to_token == to_token_address
                if to_chain_id == SOLANA_CHAIN_ID
                else action_to_token.lower() == to_token_address.lower()
            )
            token_addresses_match = from_token_matches and to_token_matches
            if not token_addresses_match:
                logger.error("LI.FI quote tokens do not match the requested route")
                return None

            resolved_to_decimals = to_token_decimals
            if resolved_to_decimals is None:
                resolved_to_decimals = int(
                    ((action.get("toToken") or {}).get("decimals")) or 18
                )
            from_amount_units = int(estimate.get("fromAmount") or amount_int)
            to_amount_units = int(estimate.get("toAmount") or 0)
            to_amount_min_units = int(
                estimate.get("toAmountMin") or to_amount_units
            )
            if to_amount_units <= 0 or to_amount_min_units <= 0:
                logger.error("LI.FI quote returned no destination amount")
                return None

            raw_price_impact = estimate.get("priceImpact")
            try:
                price_impact = (
                    float(raw_price_impact)
                    if raw_price_impact is not None
                    else None
                )
            except (TypeError, ValueError):
                price_impact = None

            return LiFiSwapQuote(
                from_chain_id=from_chain_id,
                to_chain_id=to_chain_id,
                from_token_address=from_token_address,
                to_token_address=to_token_address,
                from_amount_base_units=str(from_amount_units),
                from_amount_decimal=from_amount_units / (10 ** from_token_decimals),
                to_amount_decimal=to_amount_units / (10 ** resolved_to_decimals),
                to_token_decimals=resolved_to_decimals,
                min_to_amount_decimal=to_amount_min_units / (10 ** resolved_to_decimals),
                approval_address=estimate.get("approvalAddress"),
                transaction_request=transaction_request,
                tool_name=(
                    (quote_data.get("toolDetails") or {}).get("name")
                    or quote_data.get("tool")
                    or "Unknown"
                ),
                quote_id=quote_data.get("id", f"transfer_{int(time.time())}"),
                price_impact=price_impact,
                valid_until=datetime.now(),
                transaction_kind=transaction_kind,
                from_address=from_address,
                to_address=to_address,
                from_amount_for_gas=(
                    str(gas_amount_int) if gas_amount_int else None
                ),
            )
        except Exception as e:
            logger.error(f"Error getting LI.FI token-transfer quote: {e}")
            return None

    async def get_transfer_status(
        self,
        tx_hash: str,
        from_chain_id: int,
        to_chain_id: Optional[int] = None,
        bridge_tool: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Poll LiFi's /status endpoint for a submitted swap or bridge transaction.

        Returns the raw status payload (has a "status" field: NOT_FOUND, INVALID,
        PENDING, DONE, FAILED) or None if the request itself failed.
        """
        try:
            params: Dict[str, Any] = {
                "txHash": tx_hash,
                "fromChain": from_chain_id,
            }
            if to_chain_id is not None:
                params["toChain"] = to_chain_id
            if bridge_tool:
                params["bridge"] = bridge_tool

            response = await self.http_client.get(f"{self.base_url}/status", params=params)
            if response.status_code != 200:
                logger.warning(f"LiFi status API error: {response.status_code} - {response.text}")
                return None

            return response.json()

        except Exception as e:
            logger.error(f"Error polling LiFi transfer status: {e}")
            return None

    async def close(self):
        """Close the HTTP client."""
        try:
            await self.http_client.aclose()
            logger.info("LiFi bridge service closed")
        except Exception as e:
            logger.error(f"Error closing LiFi service: {e}")


# Global LiFi service instance
lifi_service = None


def get_lifi_service(api_key: Optional[str] = None) -> LiFiBridgeService:
    """Get or create global LiFi service instance."""
    global lifi_service

    if lifi_service is None:
        lifi_service = LiFiBridgeService(api_key=api_key)

    return lifi_service
