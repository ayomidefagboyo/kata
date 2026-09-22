"""
User Wallet Delegation Service

Manages user delegations to our authorization key for automated trading.
Implements Privy's "Authorization Key as Signer" pattern.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from dataclasses import dataclass
import json
from uuid import UUID

from ..config.database import get_service_client

logger = logging.getLogger(__name__)

def parse_timestamp(timestamp_str: str) -> Optional[datetime]:
    if not timestamp_str:
        return None
    try:
        cleaned = timestamp_str.replace('Z', '+00:00')
        if '.' in cleaned and '+' in cleaned:
            date_part, tz_part = cleaned.split('+')
            if '.' in date_part:
                base, microseconds = date_part.split('.')
                microseconds = microseconds[:6].ljust(6, '0')
                cleaned = f"{base}.{microseconds}+{tz_part}"
        dt = datetime.fromisoformat(cleaned)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception as e:
        logger.warning(f"Failed to parse timestamp '{timestamp_str}': {e}")
        return None


@dataclass
class DelegationPolicy:
    """User-defined trading policy limits."""
    max_daily_volume_usd: float = 1000.0
    max_position_size_percent: float = 10.0  # % of portfolio
    max_leverage: int = 5
    allowed_chains: List[str] = None
    require_stop_loss: bool = True
    max_trades_per_day: int = 50

    def __post_init__(self):
        if self.allowed_chains is None:
            self.allowed_chains = ['base', 'arbitrum']

    def to_dict(self) -> Dict[str, Any]:
        return {
            'max_daily_volume_usd': self.max_daily_volume_usd,
            'max_position_size_percent': self.max_position_size_percent,
            'max_leverage': self.max_leverage,
            'allowed_chains': self.allowed_chains,
            'require_stop_loss': self.require_stop_loss,
            'max_trades_per_day': self.max_trades_per_day
        }

    def covers(self, requested: 'DelegationPolicy') -> bool:
        """Return whether this policy is at least as permissive as the requested policy."""
        if float(self.max_daily_volume_usd or 0) < float(requested.max_daily_volume_usd or 0):
            return False
        if float(self.max_position_size_percent or 0) < float(requested.max_position_size_percent or 0):
            return False
        if float(self.max_leverage or 0) < float(requested.max_leverage or 0):
            return False
        if int(self.max_trades_per_day or 0) < int(requested.max_trades_per_day or 0):
            return False
        if requested.require_stop_loss and not self.require_stop_loss:
            return False

        current_chains = {
            str(chain).strip().lower()
            for chain in (self.allowed_chains or [])
            if str(chain).strip()
        }
        requested_chains = {
            str(chain).strip().lower()
            for chain in (requested.allowed_chains or [])
            if str(chain).strip()
        }
        return requested_chains.issubset(current_chains)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DelegationPolicy':
        # Ignore unknown/legacy keys (e.g. allowed_tokens from older stored records)
        known_fields = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in data.items() if key in known_fields})


@dataclass
class UserDelegation:
    """User wallet delegation record."""
    user_id: str
    wallet_id: str
    wallet_address: str
    auth_key_id: str
    chain_type: str = 'ethereum'
    delegation_type: str = 'signer'
    policy_limits: DelegationPolicy = None
    delegation_id: Optional[str] = None
    is_active: bool = True
    delegated_at: Optional[datetime] = None
    revoked_at: Optional[datetime] = None

    def __post_init__(self):
        if self.policy_limits is None:
            self.policy_limits = DelegationPolicy()
        if self.delegated_at is None:
            self.delegated_at = datetime.now()


class DelegationService:
    """
    Service for managing user wallet delegations to our authorization key.

    This service implements Privy's recommended "Authorization Key as Signer" pattern
    where users remain owners of their wallets and delegate specific trading
    permissions to our server's authorization key.
    """

    def __init__(self):
        self.supabase = get_service_client()
        self.auth_key_id = self._get_auth_key_id()
        if self.auth_key_id:
            logger.info("DelegationService initialized with configured auth key")
        else:
            logger.warning("DelegationService initialized without PRIVY_AUTHORIZATION_KEY_ID")

    def _get_auth_key_id(self) -> Optional[str]:
        """Get our authorization key ID from environment."""
        import os
        return os.getenv('PRIVY_AUTHORIZATION_KEY_ID')

    def is_configured_for_live_delegation(self) -> bool:
        """Return whether all credentials needed to create live delegation are configured."""
        import os
        return bool(
            os.getenv('PRIVY_APP_ID') and
            os.getenv('PRIVY_APP_SECRET') and
            os.getenv('PRIVY_AUTHORIZATION_KEY_ID') and
            os.getenv('PRIVY_AUTHORIZATION_KEY_PRIVATE_KEY')
        )

    @staticmethod
    def _is_uuid(value: str) -> bool:
        try:
            UUID(str(value))
            return True
        except (TypeError, ValueError):
            return False

    def _update_user_delegation_summary(
        self,
        user_id: str,
        wallet_address: Optional[str],
        status: str,
        account_wallet_address: Optional[str] = None,
    ) -> None:
        """
        Maintain a lightweight account-level delegation summary.

        user_wallet_delegations remains the authorization source of truth; these
        columns are only for fast account reads and UI state.
        """
        now = datetime.now().isoformat()
        delegated_wallet_address = wallet_address.lower() if status == "delegated" and wallet_address else None
        summary_data = {
            'delegation_status': status,
            'delegated_wallet_address': delegated_wallet_address,
            'delegation_last_checked_at': now,
            'updated_at': now
        }
        if wallet_address:
            # Keep the account row linked to the actual Privy embedded trading
            # wallet. Legacy credit-created rows copied the external login
            # wallet into platform_wallet, which made this cache impossible to
            # refresh even though the delegation source of truth was valid.
            summary_data['platform_wallet'] = wallet_address.lower()

        selectors = []
        if self._is_uuid(user_id):
            selectors.append(('id', user_id))

        selectors.extend([
            ('external_wallet', user_id),
            ('platform_wallet', user_id)
        ])

        if account_wallet_address:
            for account_wallet in {account_wallet_address, account_wallet_address.lower()}:
                selectors.extend([
                    ('external_wallet', account_wallet),
                    ('platform_wallet', account_wallet),
                ])

        if wallet_address:
            wallet_variants = {wallet_address, wallet_address.lower()}
            for wallet in wallet_variants:
                selectors.extend([
                    ('external_wallet', wallet),
                    ('platform_wallet', wallet)
                ])

        seen_selectors = set()
        for column, value in selectors:
            if not value:
                continue

            selector_key = (column, value)
            if selector_key in seen_selectors:
                continue

            seen_selectors.add(selector_key)
            try:
                self.supabase.table('users').update(summary_data).eq(column, value).execute()
            except Exception as e:
                logger.warning(
                    f"Could not update delegation summary using users.{column}={value}: {e}"
                )

    async def store_delegation(
        self,
        user_id: str,
        wallet_id: str,
        wallet_address: str,
        policy_limits: DelegationPolicy,
        chain_type: str = 'ethereum',
        delegation_id: Optional[str] = None,
        account_wallet_address: Optional[str] = None,
    ) -> UserDelegation:
        """
        Store a user's delegation to our authorization key.

        Args:
            user_id: Privy user ID
            wallet_id: User's wallet ID
            wallet_address: User's wallet address
            policy_limits: User-defined trading limits
            delegation_id: Privy delegation ID (if available)

        Returns:
            UserDelegation record
        """
        try:
            if not self.auth_key_id:
                raise ValueError("PRIVY_AUTHORIZATION_KEY_ID must be configured before storing delegation")

            delegation = UserDelegation(
                user_id=user_id,
                wallet_id=wallet_id,
                wallet_address=wallet_address,
                auth_key_id=self.auth_key_id,
                chain_type=chain_type,
                policy_limits=policy_limits,
                delegation_id=delegation_id
            )

            # Store in database
            delegation_data = {
                'user_id': delegation.user_id,
                'wallet_id': delegation.wallet_id,
                'wallet_address': delegation.wallet_address,
                'chain_type': delegation.chain_type,
                'delegation_id': delegation.delegation_id,
                'auth_key_id': delegation.auth_key_id,
                'delegation_type': delegation.delegation_type,
                'policy_limits': delegation.policy_limits.to_dict(),
                'is_active': delegation.is_active,
                'delegated_at': delegation.delegated_at.isoformat()
            }

            result = self.supabase.table('user_wallet_delegations').upsert(
                delegation_data,
                on_conflict='user_id,wallet_id'
            ).execute()

            if result.data:
                if chain_type == "ethereum":
                    self._update_user_delegation_summary(
                        user_id,
                        wallet_address,
                        "delegated",
                        account_wallet_address=account_wallet_address,
                    )
                logger.info(f"Stored delegation for user {user_id} with wallet {wallet_id}")
                return delegation
            else:
                raise Exception("Failed to store delegation in database")

        except Exception as e:
            logger.error(f"Error storing delegation: {e}")
            raise

    async def get_user_delegation(
        self,
        user_id: str,
        chain_type: str = 'ethereum',
    ) -> Optional[UserDelegation]:
        """
        Get active delegation for a user.

        Args:
            user_id: Privy user ID

        Returns:
            UserDelegation if active delegation exists, None otherwise
        """
        try:
            result = self.supabase.table('user_wallet_delegations').select('*').eq(
                'user_id', user_id
            ).eq('chain_type', chain_type).eq('is_active', True).execute()

            if result.data and len(result.data) > 0:
                data = result.data[0]
                delegation = UserDelegation(
                    user_id=data['user_id'],
                    wallet_id=data['wallet_id'],
                    wallet_address=data['wallet_address'],
                    auth_key_id=data['auth_key_id'],
                    chain_type=data.get('chain_type') or 'ethereum',
                    delegation_type=data['delegation_type'],
                    policy_limits=DelegationPolicy.from_dict(data['policy_limits']),
                    delegation_id=data.get('delegation_id'),
                    is_active=data['is_active'],
                    delegated_at=parse_timestamp(data['delegated_at']),
                    revoked_at=parse_timestamp(data['revoked_at']) if data.get('revoked_at') else None
                )
                return delegation

            return None

        except Exception as e:
            logger.error(f"Error getting user delegation: {e}")
            return None

    async def get_user_delegation_and_refresh_summary(
        self,
        user_id: str,
        account_wallet_address: Optional[str] = None,
        chain_type: str = 'ethereum',
    ) -> Optional[UserDelegation]:
        """
        Get active delegation and refresh the account-level cache.

        The returned delegation is still read from user_wallet_delegations, which
        remains the source of truth for trading authorization.
        """
        delegation = await self.get_user_delegation(user_id, chain_type=chain_type)
        if chain_type == "ethereum":
            if delegation:
                self._update_user_delegation_summary(
                    user_id,
                    delegation.wallet_address,
                    "delegated",
                    account_wallet_address=account_wallet_address,
                )
            else:
                self._update_user_delegation_summary(
                    user_id,
                    None,
                    "not_delegated",
                    account_wallet_address=account_wallet_address,
                )

        return delegation

    async def revoke_delegation(self, user_id: str) -> bool:
        """
        Revoke a user's delegation.

        Args:
            user_id: Privy user ID

        Returns:
            True if revocation successful, False otherwise
        """
        try:
            existing_delegation = await self.get_user_delegation(user_id)

            result = self.supabase.table('user_wallet_delegations').update({
                'is_active': False,
                'revoked_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }).eq('user_id', user_id).eq('is_active', True).execute()

            if result.data:
                wallet_address = (
                    existing_delegation.wallet_address
                    if existing_delegation
                    else result.data[0].get('wallet_address')
                )
                self._update_user_delegation_summary(user_id, wallet_address, "revoked")
                logger.info(f"Revoked delegation for user {user_id}")
                return True
            else:
                logger.warning(f"No active delegation found for user {user_id}")
                return False

        except Exception as e:
            logger.error(f"Error revoking delegation: {e}")
            return False

    async def has_delegation(self, user_id: str, chain_type: str = 'ethereum') -> bool:
        """
        Check if user has active delegation.

        Args:
            user_id: Privy user ID

        Returns:
            True if user has active delegation, False otherwise
        """
        delegation = await self.get_user_delegation(user_id, chain_type=chain_type)
        return delegation is not None and delegation.is_active

    async def validate_trade_against_policy(
        self,
        user_id: str,
        trade_data: Dict[str, Any]
    ) -> tuple[bool, Optional[str]]:
        """
        Validate a trade against user's delegation policy.

        Args:
            user_id: Privy user ID
            trade_data: Trade details

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            chain_type = str(trade_data.get('chain_type') or 'ethereum').lower()
            delegation = await self.get_user_delegation(user_id, chain_type=chain_type)
            if not delegation:
                return False, f"No active {chain_type} delegation found for user"

            policy = delegation.policy_limits

            chain = str(trade_data.get('chain') or '').lower()
            if chain and chain not in {item.lower() for item in policy.allowed_chains}:
                return False, f"Chain {chain} is not allowed by the user's delegation policy"

            # Validate volume limit
            trade_volume = trade_data.get('volume_usd', 0)
            if trade_volume > policy.max_daily_volume_usd:
                return False, f"Trade volume ${trade_volume} exceeds daily limit ${policy.max_daily_volume_usd}"

            # Validate position size
            position_size_percent = trade_data.get('position_size_percent', 0)
            if position_size_percent > policy.max_position_size_percent:
                return False, f"Position size {position_size_percent}% exceeds limit {policy.max_position_size_percent}%"

            # Validate leverage
            leverage = trade_data.get('leverage', 1)
            if leverage > policy.max_leverage:
                return False, f"Leverage {leverage}x exceeds limit {policy.max_leverage}x"

            # No token allowlist: Yuki trades the full venue-listed universe
            # (platform signals span every Hyperliquid perp). Volume, size,
            # leverage, and stop-loss limits above remain the guardrails.

            # Validate stop loss requirement
            if policy.require_stop_loss and not trade_data.get('stop_loss'):
                return False, "Stop loss is required by user policy"

            return True, None

        except Exception as e:
            logger.error(f"Error validating trade against policy: {e}")
            return False, f"Policy validation error: {str(e)}"

    async def get_all_active_delegations(self) -> List[UserDelegation]:
        """
        Get all active delegations.

        Returns:
            List of active UserDelegation records
        """
        try:
            result = self.supabase.table('user_wallet_delegations').select('*').eq(
                'is_active', True
            ).execute()

            delegations = []
            for data in result.data:
                delegation = UserDelegation(
                    user_id=data['user_id'],
                    wallet_id=data['wallet_id'],
                    wallet_address=data['wallet_address'],
                    auth_key_id=data['auth_key_id'],
                    chain_type=data.get('chain_type') or 'ethereum',
                    delegation_type=data['delegation_type'],
                    policy_limits=DelegationPolicy.from_dict(data['policy_limits']),
                    delegation_id=data.get('delegation_id'),
                    is_active=data['is_active'],
                    delegated_at=parse_timestamp(data['delegated_at']),
                    revoked_at=parse_timestamp(data['revoked_at']) if data.get('revoked_at') else None
                )
                delegations.append(delegation)

            return delegations

        except Exception as e:
            logger.error(f"Error getting all active delegations: {e}")
            return []


# Global service instance
_delegation_service = None


def get_delegation_service() -> DelegationService:
    """Get global delegation service instance."""
    global _delegation_service
    if _delegation_service is None:
        _delegation_service = DelegationService()
    return _delegation_service
