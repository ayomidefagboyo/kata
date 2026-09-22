"""
Hyperliquid Configuration Management

Configuration settings and environment management for Hyperliquid integration.
Includes API credentials, risk parameters, and trading settings.
"""

import os
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class HyperliquidConfig:
    """Hyperliquid configuration settings."""
    
    # API Configuration
    api_key: str
    api_secret: str
    testnet: bool = False
    
    # Base URLs
    base_url: str = ""
    info_url: str = ""
    exchange_url: str = ""
    
    # Risk Management Parameters
    max_leverage: float = 10.0
    max_position_size_usd: float = 50000.0
    margin_buffer_percent: float = 30.0
    liquidation_warning_threshold: float = 0.15
    max_correlation_exposure: float = 0.6
    funding_rate_threshold: float = 0.01
    max_daily_funding_cost: float = 1000.0
    
    # Trading Parameters
    default_leverage: float = 2.0
    min_order_size_usd: float = 10.0
    max_slippage_percent: float = 0.5
    order_timeout_seconds: int = 30
    
    # Cache Settings
    market_data_cache_duration: int = 5
    position_cache_duration: int = 10
    
    # Rate Limiting
    max_requests_per_second: int = 10
    burst_limit: int = 20
    
    def __post_init__(self):
        """Set derived URLs based on testnet setting."""
        if self.testnet:
            self.base_url = "https://api.hyperliquid-testnet.xyz"
        else:
            self.base_url = "https://api.hyperliquid.xyz"
        
        self.info_url = f"{self.base_url}/info"
        self.exchange_url = f"{self.base_url}/exchange"


def load_hyperliquid_config() -> HyperliquidConfig:
    """
    Load Hyperliquid configuration from environment variables.
    
    Required environment variables:
    - HYPERLIQUID_API_KEY: Your Hyperliquid API key
    - HYPERLIQUID_API_SECRET: Your Hyperliquid API secret
    
    Optional environment variables:
    - HYPERLIQUID_TESTNET: Use testnet (default: true)
    - HYPERLIQUID_MAX_LEVERAGE: Maximum leverage allowed (default: 10.0)
    - HYPERLIQUID_MAX_POSITION_SIZE: Maximum position size in USD (default: 50000.0)
    """
    
    # Required settings
    api_key = os.getenv('HYPERLIQUID_API_KEY')
    api_secret = os.getenv('HYPERLIQUID_API_SECRET')
    
    if not api_key or not api_secret:
        raise ValueError("HYPERLIQUID_API_KEY and HYPERLIQUID_API_SECRET environment variables are required")
    
    # Optional settings with defaults
    testnet = os.getenv('HYPERLIQUID_TESTNET', 'true').lower() == 'true'
    max_leverage = float(os.getenv('HYPERLIQUID_MAX_LEVERAGE', '10.0'))
    max_position_size = float(os.getenv('HYPERLIQUID_MAX_POSITION_SIZE', '50000.0'))
    margin_buffer = float(os.getenv('HYPERLIQUID_MARGIN_BUFFER', '30.0'))
    liquidation_warning = float(os.getenv('HYPERLIQUID_LIQUIDATION_WARNING', '0.15'))
    max_correlation = float(os.getenv('HYPERLIQUID_MAX_CORRELATION', '0.6'))
    funding_threshold = float(os.getenv('HYPERLIQUID_FUNDING_THRESHOLD', '0.01'))
    max_funding_cost = float(os.getenv('HYPERLIQUID_MAX_FUNDING_COST', '1000.0'))
    
    return HyperliquidConfig(
        api_key=api_key,
        api_secret=api_secret,
        testnet=testnet,
        max_leverage=max_leverage,
        max_position_size_usd=max_position_size,
        margin_buffer_percent=margin_buffer,
        liquidation_warning_threshold=liquidation_warning,
        max_correlation_exposure=max_correlation,
        funding_rate_threshold=funding_threshold,
        max_daily_funding_cost=max_funding_cost
    )


# Risk profiles for different trading strategies
RISK_PROFILES = {
    'conservative': {
        'max_leverage': 3.0,
        'max_position_size_usd': 10000.0,
        'margin_buffer_percent': 50.0,
        'liquidation_warning_threshold': 0.3,
        'max_correlation_exposure': 0.3,
        'funding_rate_threshold': 0.005,
        'max_daily_funding_cost': 100.0
    },
    'moderate': {
        'max_leverage': 5.0,
        'max_position_size_usd': 25000.0,
        'margin_buffer_percent': 40.0,
        'liquidation_warning_threshold': 0.25,
        'max_correlation_exposure': 0.5,
        'funding_rate_threshold': 0.0075,
        'max_daily_funding_cost': 500.0
    },
    'aggressive': {
        'max_leverage': 10.0,
        'max_position_size_usd': 50000.0,
        'margin_buffer_percent': 30.0,
        'liquidation_warning_threshold': 0.15,
        'max_correlation_exposure': 0.7,
        'funding_rate_threshold': 0.01,
        'max_daily_funding_cost': 1000.0
    },
    'yuki_futures': {  # Specialized for Yuki agent
        'max_leverage': 10.0,
        'max_position_size_usd': 75000.0,
        'margin_buffer_percent': 25.0,
        'liquidation_warning_threshold': 0.12,
        'max_correlation_exposure': 0.8,
        'funding_rate_threshold': 0.015,
        'max_daily_funding_cost': 2000.0
    }
}


def get_risk_profile(profile_name: str) -> Dict[str, Any]:
    """Get risk profile configuration by name."""
    if profile_name not in RISK_PROFILES:
        raise ValueError(f"Unknown risk profile: {profile_name}. Available: {list(RISK_PROFILES.keys())}")
    
    return RISK_PROFILES[profile_name]


def apply_risk_profile(config: HyperliquidConfig, profile_name: str) -> HyperliquidConfig:
    """Apply a risk profile to the configuration."""
    profile = get_risk_profile(profile_name)
    
    # Update config with profile settings
    for key, value in profile.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config


# Contract specifications for supported futures
SUPPORTED_CONTRACTS = {
    'BTC-PERP': {
        'tick_size': 1.0,
        'min_size': 0.001,
        'max_leverage': 50,
        'maintenance_margin': 0.005,  # 0.5%
        'initial_margin': 0.02,       # 2%
        'asset_class': 'major_crypto',
        'correlation_group': 'btc_ecosystem'
    },
    'ETH-PERP': {
        'tick_size': 0.1,
        'min_size': 0.01,
        'max_leverage': 50,
        'maintenance_margin': 0.005,
        'initial_margin': 0.02,
        'asset_class': 'major_crypto',
        'correlation_group': 'eth_ecosystem'
    },
    'SOL-PERP': {
        'tick_size': 0.01,
        'min_size': 0.1,
        'max_leverage': 20,
        'maintenance_margin': 0.0075,
        'initial_margin': 0.05,
        'asset_class': 'layer1_alt',
        'correlation_group': 'sol_ecosystem'
    },
    'AVAX-PERP': {
        'tick_size': 0.01,
        'min_size': 0.1,
        'max_leverage': 20,
        'maintenance_margin': 0.01,
        'initial_margin': 0.05,
        'asset_class': 'layer1_alt',
        'correlation_group': 'avax_ecosystem'
    },
    'MATIC-PERP': {
        'tick_size': 0.0001,
        'min_size': 1.0,
        'max_leverage': 20,
        'maintenance_margin': 0.01,
        'initial_margin': 0.05,
        'asset_class': 'layer2_scaling',
        'correlation_group': 'eth_ecosystem'
    }
}


def get_contract_spec(symbol: str) -> Dict[str, Any]:
    """Get contract specifications for a symbol."""
    if symbol not in SUPPORTED_CONTRACTS:
        raise ValueError(f"Unsupported contract: {symbol}")
    
    return SUPPORTED_CONTRACTS[symbol]


def validate_config(config: HyperliquidConfig) -> Dict[str, Any]:
    """
    Validate Hyperliquid configuration settings.
    
    Returns:
        Dictionary with validation results
    """
    issues = []
    warnings = []
    
    # Check API credentials
    if not config.api_key or len(config.api_key) < 10:
        issues.append("Invalid or missing API key")
    
    if not config.api_secret or len(config.api_secret) < 10:
        issues.append("Invalid or missing API secret")
    
    # Check risk parameters
    if config.max_leverage > 50:
        warnings.append("Very high maximum leverage - consider reducing for safety")
    
    if config.margin_buffer_percent < 20:
        warnings.append("Low margin buffer - increase for better risk management")
    
    if config.liquidation_warning_threshold < 0.1:
        warnings.append("Very low liquidation warning threshold - increase for early warnings")
    
    if config.max_position_size_usd > 100000:
        warnings.append("Very large maximum position size - ensure adequate risk management")
    
    # Check cache settings
    if config.market_data_cache_duration > 30:
        warnings.append("Long cache duration may result in stale market data")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'warnings': warnings
    }


# Example configuration for testing
def get_test_config() -> HyperliquidConfig:
    """Get test configuration with mock credentials."""
    return HyperliquidConfig(
        api_key="test_api_key_12345",
        api_secret="test_api_secret_67890",
        testnet=False,
        max_leverage=5.0,
        max_position_size_usd=10000.0,
        margin_buffer_percent=40.0
    )