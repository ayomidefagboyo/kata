"""
Configuration settings for the Flow AI Trading Platform backend.
"""
import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Server configuration
    HOST: str = Field(default="0.0.0.0", description="Server host")
    PORT: int = Field(default=8000, description="Server port")
    DEBUG: bool = Field(default=False, description="Debug mode")
    
    # Supabase configuration
    SUPABASE_URL: str = Field(default="https://placeholder.supabase.co", description="Supabase project URL")
    SUPABASE_ANON_KEY: str = Field(default="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJyb2xlIjoiYW5vbiIsImlzcyI6InN1cGFiYXNlIn0.placeholder", description="Supabase anonymous key")
    SUPABASE_SERVICE_KEY: str = Field(default="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJyb2xlIjoic2VydmljZV9yb2xlIiwiaXNzIjoic3VwYWJhc2UifQ.placeholder", description="Supabase service role key")
    DATABASE_URL: Optional[str] = Field(default=None, description="Direct Postgres database connection URL")
    
    # CORS configuration
    FRONTEND_URL: str = Field(default="http://localhost:5173", description="Frontend URL for CORS")
    
    # JWT configuration (optional)
    JWT_SECRET_KEY: str = Field(default="your-secret-key-here", description="JWT secret key")
    JWT_ALGORITHM: str = Field(default="HS256", description="JWT algorithm")
    JWT_EXPIRE_MINUTES: int = Field(default=60, description="JWT expiration time in minutes")
    
    # API configuration
    API_V1_PREFIX: str = "/api"
    PROJECT_NAME: str = "Kata Autonomous Perpetual Agent"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "Autonomous perpetual futures agent with closed-loop learning"

    # Execution Venue configuration
    DEFAULT_VENUE: str = Field(default="hyperliquid", description="Default perpetual futures execution venue (hyperliquid, drift)")
    
    # Privy configuration (optional for standalone / direct key execution)
    PRIVY_APP_ID: Optional[str] = Field(default="", description="Privy App ID")
    PRIVY_APP_SECRET: Optional[str] = Field(default="", description="Privy App Secret")
    
    # Market Data API configuration
    COINGECKO_API_KEY: Optional[str] = Field(default=None, description="CoinGecko Pro API key (optional)")
    COINGECKO_API_KEYS: Optional[str] = Field(default=None, description="Comma-separated list of CoinGecko API keys for rotation")
    LIFI_API_KEY: Optional[str] = Field(default=None, description="LI.FI API key for Ryu funding and spot routes")
    COINGECKO_BASE_URL_FREE: str = Field(default="https://api.coingecko.com/api/v3", description="Base URL for free tier")
    COINGECKO_BASE_URL_PRO: str = Field(default="https://pro-api.coingecko.com/api/v3", description="Base URL for Pro tier")
    # Free-tier throttling knobs
    COINGECKO_FREE_RATE_CALLS_PER_MINUTE: int = Field(default=15, description="Max calls/min for free tier")
    COINGECKO_FREE_MIN_DELAY_SECONDS: float = Field(default=4.5, description="Min seconds between requests")
    COINGECKO_FREE_PAGE_DELAY_SECONDS: float = Field(default=12.0, description="Delay between discovery pages")
    COINGECKO_FREE_ERROR_BACKOFF_BASE: int = Field(default=60, description="Base backoff for 429/403 in seconds")
    COINGECKO_FREE_ERROR_BACKOFF_MAX: int = Field(default=600, description="Max backoff in seconds")
    BINANCE_API_KEY: Optional[str] = Field(default=None, description="Binance API key (optional)")
    BINANCE_API_SECRET: Optional[str] = Field(default=None, description="Binance API secret (optional)")
    MARKET_DATA_TESTNET: bool = Field(default=True, description="Use testnet for market data APIs")
    OPPORTUNITY_MIN_VOLUME_USDT: float = Field(
        default=20_000_000.0,
        description="Minimum 24h quote volume (USDT) for opportunity scoring eligibility",
    )
    DISCOVERY_DEPTH_MIN_SIDE_USDT: float = Field(
        default=100_000.0,
        description="Minimum L2 depth (USDT) required on both bid and ask sides during discovery",
    )
    DISCOVERY_DEPTH_MAX_SPREAD_PCT: float = Field(
        default=0.25,
        description="Maximum allowed top-of-book spread percentage in discovery depth gate",
    )
    SIGNAL_MIN_SL_DISTANCE_PCT: float = Field(
        default=0.015,
        description="Minimum stop-loss distance as a fraction of entry price (e.g. 0.015 = 1.5%)",
    )
    SIGNAL_MIN_NET_RR_THRESHOLD: float = Field(
        default=1.15,
        description="Minimum net risk-reward ratio after costs required to pass final guardrail",
    )
    VALIDITY_CALIBRATION_MAX_MULTIPLIER: float = Field(
        default=3.0,
        description="Maximum multiplier over canonical horizon used when calibrating validity windows from historical outcomes",
    )

    # --- Expiry control (P0 fix: ~61% of signals were expiring unresolved) ---
    # Floor + buffer applied to the PUBLISHED validity window so short AI horizons
    # still get enough time to reach target/stop instead of expiring as noise.
    SIGNAL_MIN_VALIDITY_HOURS: float = Field(
        default=24.0,
        description="Hard floor on published signal validity window (hours). Stops short-horizon signals expiring before they can resolve.",
    )
    SIGNAL_VALIDITY_BUFFER_MULT: float = Field(
        default=1.5,
        description="Multiplier applied to the horizon-derived validity window when publishing, giving price room to reach target/stop.",
    )
    SIGNAL_MAX_VALIDITY_HOURS: float = Field(
        default=336.0,
        description="Cap on published validity window (hours) so signals still resolve in a bounded time (default 2 weeks).",
    )

    # --- Publish confidence floor ---
    # Data-backed: signals < 0.60 lose or break even in every regime cut (26-31% WR,
    # flat/negative expectancy). The [0.60,0.65) band is the largest bucket (n=192) and
    # is modestly profitable (55.7% WR, +1.18%/trade), so we keep it rather than blunt-
    # cutting at 0.65. A 0.65 cut leaned too hard on October (85% of >=0.65 samples).
    SIGNAL_PUBLISH_CONFIDENCE_FLOOR: float = Field(
        default=0.60,
        description="Absolute minimum confidence to publish a directional signal. Below ~0.60 historical win rate is sub-coin-flip with flat/negative expectancy.",
    )
    # Signals at/above this are flagged HIGH_CONVICTION: the [0.70,0.80) band shows the
    # best realized performance (75.5% WR, +5.54%/trade). Used to tag a premium feed and
    # (optionally) scale position size up within existing risk limits.
    SIGNAL_HIGH_CONVICTION_THRESHOLD: float = Field(
        default=0.70,
        description="Confidence at/above which a signal is tagged high-conviction (premium feed / larger size within risk limits).",
    )
    # NEUTRALISED 2026-06-23: June data showed confidence calibration INVERTED
    # (conf>=0.70 won 20% vs <0.70 won 30%), so a >1.0 uplift was upsizing the worst
    # bucket. Hold at 1.0 until calibration is re-validated on fresh, resolved data.
    SIGNAL_HIGH_CONVICTION_SIZE_MULT: float = Field(
        default=1.0,
        description="Position-size multiplier for high-conviction signals. Held at 1.0 (neutral) until confidence calibration is re-validated; raise only when conf>=0.70 demonstrably outperforms on recent resolved data.",
    )

    # --- HTF trend-alignment gate (extreme-only safety net) ---
    # NOT a short-blocker. Only vetoes signals fighting a VERY strong daily trend (both
    # distance AND slope extreme). Normal direction selectivity is handled adaptively by
    # the directional-learning haircut below, which self-corrects from recent outcomes.
    HTF_TREND_GATE_DISTANCE_PCT: float = Field(
        default=3.0,
        description="Daily price-vs-EMA20 distance (%); with slope, defines a VERY strong trend that vetoes counter-trend signals.",
    )
    HTF_TREND_GATE_SLOPE_PCT: float = Field(
        default=1.0,
        description="Daily EMA5-vs-EMA20 slope (%); with distance, defines a VERY strong trend that vetoes counter-trend signals.",
    )

    # --- Directional self-learning (2026-06-23) ---
    # The system reads recent resolved win-rate BY DIRECTION and automatically haircuts
    # confidence for whichever side has been failing (e.g. "shorts haven't worked lately"),
    # then reverses on its own when that side recovers. This replaces a static short-block
    # with an adaptive, data-driven bias that rides the regime.
    DIRECTION_LEARNING_ENABLED: bool = Field(
        default=True,
        description="Enable adaptive per-direction confidence adjustment from recent resolved outcomes.",
    )
    DIRECTION_LEARNING_LOOKBACK_DAYS: int = Field(
        default=30,
        description="Lookback window (days) for recent per-direction win-rate. 30d balances recency against current low volume (14d is too sparse to be meaningful); the adjustment auto-scales by sample size regardless.",
    )
    DIRECTION_LEARNING_MIN_SAMPLES: int = Field(
        default=8,
        description="Resolved signals per direction before the adjustment reaches full strength; below this it is scaled down.",
    )
    DIRECTION_LEARNING_MAX_CONF_ADJ: float = Field(
        default=0.10,
        description="Maximum absolute confidence adjustment (±) applied from directional learning.",
    )

    # --- Gate removal (2026-06-23) ---
    # June ran a 3% publish rate / 6-day drought because a stack of hard pre-LLM vetoes
    # rejected ~97% of candidates. These gates overlap with the adaptive confidence floor +
    # directional learning, which now carry quality control. Disable the redundant binary
    # vetoes (set any True to re-enable). Kept always-on: drawdown breaker, HOLD/non-
    # directional, data-quality, confidence floor, RR floor, extreme-only trend gate.
    GATE_ENABLE_CROWDING: bool = Field(default=False, description="Veto signals when positioning is crowded in the trade direction.")
    GATE_ENABLE_NEGATIVE_EV: bool = Field(default=False, description="Veto signals with historically negative expected value (pre-LLM).")
    GATE_ENABLE_SCOPED_LEARNING_VETO: bool = Field(default=False, description="Allow scoped-learning context to hard-veto before the LLM runs.")
    GATE_ENABLE_POLICY_PRE_LLM_VETO: bool = Field(default=False, description="Veto before the LLM when offline policy P(win) is low. Off = let the LLM still analyse.")
    GATE_ENABLE_ATTRIBUTION_COMBO: bool = Field(default=False, description="Veto signals whose feature combo is historically underperforming.")
    GATE_ENABLE_ENSEMBLE_DISAGREEMENT: bool = Field(default=False, description="Veto when AI and rule-engine direction/confidence disagree.")
    GATE_ENABLE_POLICY_POST_AI: bool = Field(default=False, description="Veto after AI when offline policy P(win) is below floor. Off = confidence floor governs.")

    # --- Token analysis (spot BUY/SELL/HOLD) tuning (2026-06-23) ---
    # The combiner needed |combined_score| >= 0.22 to call BUY/SELL, so most setups fell
    # into the HOLD dead zone (worsened when the LLM returns HOLD and halves the score).
    # Lower the threshold to be more decisive; entries are also made direction-aware so a
    # BUY no longer defaults to "buy at/above current price".
    TOKEN_ANALYSIS_DECISION_THRESHOLD: float = Field(
        default=0.15,
        description="Absolute combined-direction score needed to call BUY/SELL in token analysis. Lower = fewer HOLDs.",
    )
    TOKEN_ANALYSIS_SOLO_DECISION_CONFIDENCE: float = Field(
        default=0.72,
        description="Minimum confidence for one directional vote to override a HOLD vote in token analysis.",
    )

    # --- Rule-engine macro trend alignment (2026-06-23) ---
    # The rule engine is a mean-reversion fader: 'overbought_reversal' (SHORT) scores
    # highest (0.8), so it reflexively SHORTed strength → 74% short candidates even when
    # the broad market was rising. Fix: when the BTC MACRO regime is strongly trending
    # (the same regime token analysis uses), gently DISCOUNT counter-trend setups. We do
    # NOT boost with-trend setups (default boost 1.0) — the market decides direction and
    # mean reversion stays valid; in ranging/sideways regimes no weighting is applied.
    RULE_TREND_ALIGN_ENABLED: bool = Field(default=True, description="Discount counter-trend rule setups when the BTC macro regime is strongly trending.")
    RULE_TREND_ALIGN_BOOST: float = Field(default=1.0, description="Score multiplier for with-trend setups. Default 1.0 = no boost, so no manufactured long/trend bias.")
    RULE_TREND_ALIGN_PENALTY: float = Field(default=0.80, description="Score multiplier for counter-trend setups in a strong macro trend. Mean reversion still allowed, just deprioritised vs fighting a confirmed BTC trend.")

    # --- Learning loop anti-starvation + staleness self-heal (2026-06-23) ---
    # At ~30 signals/month, a 30-day window split by direction*confidence*leverage never
    # cleared the 15-sample floor, so pattern_performance_tracking froze. Widen the
    # window, lower the per-pattern floor (generator already down-weights small-N), and
    # add an all-history fallback when the table goes stale.
    LEARNING_PATTERN_LOOKBACK_HOURS: int = Field(
        default=24 * 180,
        description="Rolling lookback (hours) for pattern learning. Wide enough to accumulate resolved outcomes at low signal volume.",
    )
    LEARNING_MIN_PATTERN_SAMPLE: int = Field(
        default=8,
        description="Minimum resolved signals per pattern group before it is persisted. Low-N patterns are down-weighted, not trusted, by the generator.",
    )
    LEARNING_STALENESS_HOURS: float = Field(
        default=36.0,
        description="If pattern_performance_tracking has not updated within this many hours, the updater falls back to an all-history backfill instead of bailing.",
    )


    
    # Phase 2: News & Sentiment API configuration
    
    # NewsAPI configuration
    NEWSAPI_API_KEY: Optional[str] = Field(default=None, description="NewsAPI.org API key for financial news")
    
    # Reddit API configuration
    REDDIT_CLIENT_ID: Optional[str] = Field(default=None, description="Reddit API client ID")
    REDDIT_CLIENT_SECRET: Optional[str] = Field(default=None, description="Reddit API client secret")
    REDDIT_USER_AGENT: str = Field(default="YukiAgent/1.0", description="Reddit API user agent")
    
    # Twitter/X API configuration
    TWITTER_API_KEY: Optional[str] = Field(default=None, description="Twitter API key")
    TWITTER_API_SECRET: Optional[str] = Field(default=None, description="Twitter API secret")
    TWITTER_ACCESS_TOKEN: Optional[str] = Field(default=None, description="Twitter access token")
    TWITTER_ACCESS_TOKEN_SECRET: Optional[str] = Field(default=None, description="Twitter access token secret")
    TWITTER_BEARER_TOKEN: Optional[str] = Field(default=None, description="Twitter Bearer token")
    
    # FRED API configuration
    FRED_API_KEY: Optional[str] = Field(default=None, description="Federal Reserve Economic Data API key")
    
    # Hyperliquid Settings (Optional - with defaults)
    HYPERLIQUID_TESTNET: bool = Field(default=True, description="Use Hyperliquid testnet")
    HYPERLIQUID_MAX_LEVERAGE: float = Field(default=10.0, description="Maximum leverage allowed")
    HYPERLIQUID_MAX_POSITION_SIZE: float = Field(default=50000.0, description="Maximum position size in USD")
    HYPERLIQUID_MARGIN_BUFFER: float = Field(default=30.0, description="Margin buffer percentage")
    HYPERLIQUID_LIQUIDATION_WARNING: float = Field(default=0.15, description="Liquidation warning threshold")
    HYPERLIQUID_MAX_CORRELATION: float = Field(default=0.6, description="Maximum position correlation")
    HYPERLIQUID_FUNDING_THRESHOLD: float = Field(default=0.01, description="Funding rate threshold")
    HYPERLIQUID_MAX_FUNDING_COST: float = Field(default=1000.0, description="Maximum funding cost per day")
    HYPERLIQUID_WS_DETAIL_STREAMS_ENABLED: bool = Field(
        default=False,
        description="Subscribe to per-symbol Hyperliquid candle/orderbook/trade streams. Disabled by default for lean Yuki execution.",
    )
    HYPERLIQUID_HIP3_DEX_ALLOWLIST: str = Field(
        default="xyz",
        description="Comma-separated HIP-3 DEXes Yuki discovers for venue-agnostic signals. Main Hyperliquid is always included.",
    )
    YUKI_LIVE_TRADING_ENABLED: bool = Field(
        default=False,
        description="Enable Yuki to place live Hyperliquid orders. Default false until wallet signing and funding are verified.",
    )
    YUKI_REQUIRE_DELEGATION: bool = Field(
        default=True,
        description="Require active user wallet delegation before Yuki auto-trading can start.",
    )
    YUKI_MIN_ALLOCATION_USDC: float = Field(default=5.0, description="Minimum USDC allocation for Yuki.")
    YUKI_MIN_TRADE_USDC: float = Field(default=5.0, description="Minimum USDC margin amount per Yuki trade.")
    # --- Ryu (spot trading specialist) live execution ---
    RYU_LIVE_TRADING_ENABLED: bool = Field(
        default=True,
        description="Enable Ryu to place live Solana spot orders. Default true for mainnet.",
    )
    RYU_BACKGROUND_DISCOVERY_ENABLED: bool = Field(
        default=True,
        description="Enable automated background scanning for Ryu token discovery.",
    )
    RYU_MIN_ALLOCATION_USDC: float = Field(default=10.0, description="Minimum USDC allocation for Ryu.")
    RYU_SPOT_MIN_TRADE_USDC: float = Field(
        default=2.0, description="Minimum USDC value per Ryu spot buy/exit swap."
    )
    RYU_SPOT_MAX_TRADE_USDC: float = Field(
        default=200.0,
        description="Absolute ceiling per Ryu spot swap regardless of LLM-derived position sizing. "
        "Conservative default while live spot execution is new.",
    )
    RYU_SPOT_SLIPPAGE_TOLERANCE: float = Field(
        default=0.03, description="Max acceptable slippage fraction for Ryu LiFi swaps (0.03 = 3%)."
    )
    RYU_CANDIDATE_COUNT: int = Field(
        default=8,
        description="Number of ranked discovery candidates Ryu analyzes per allocation per cycle. "
        "Bounds LLM/API cost per cycle.",
    )
    RYU_MAX_CONCURRENT_POSITIONS: int = Field(
        default=5, description="Maximum open spot positions per Ryu allocation at once."
    )
    RYU_MIN_CONFIDENCE_TO_TRADE: float = Field(
        default=0.60,
        description="Minimum LLM confidence required for a live Ryu SPOT_BUY or SPOT_EXIT to execute.",
    )
    RYU_MAX_PRICE_IMPACT: float = Field(
        default=0.05,
        description="Reject Ryu routes whose estimated price impact exceeds this fraction.",
    )
    RYU_REQUIRE_RISK_PLAN: bool = Field(
        default=True,
        description="Require every new Ryu spot position to have a valid stop and at least one upside target.",
    )
    RYU_POSITION_MONITOR_ENABLED: bool = Field(
        default=True,
        description="Monitor live Ryu holdings between LLM cycles and execute persisted risk plans.",
    )
    RYU_TARGET_1_SELL_FRACTION: float = Field(
        default=0.40,
        description="Fraction of the original Ryu position sold when target 1 is reached.",
    )
    RYU_TARGET_2_SELL_FRACTION: float = Field(
        default=0.30,
        description="Fraction of the original Ryu position sold when target 2 is reached.",
    )
    RYU_TRAILING_STOP_PERCENT: float = Field(
        default=0.08,
        description="Trailing-stop distance for the Ryu runner after staged profit taking begins.",
    )
    RYU_REBALANCE_ENABLED: bool = Field(
        default=True,
        description="Allow Ryu to replace its weakest holding with a materially stronger spot opportunity.",
    )
    RYU_REBALANCE_MIN_SCORE_DELTA: float = Field(
        default=0.12,
        description="Minimum 0-1 conviction-score advantage required before Ryu rotates a holding.",
    )
    RYU_REBALANCE_MIN_HOLD_MINUTES: int = Field(
        default=360,
        description="Minimum holding time before a Ryu position can be replaced for portfolio rebalancing.",
    )
    RYU_MAX_ROTATIONS_PER_CYCLE: int = Field(
        default=1,
        description="Maximum number of weakest-position replacements in one Ryu analysis cycle.",
    )
    RYU_BASE_MIN_GAS_RESERVE_ETH: float = Field(
        default=0.0001,
        description=(
            "Minimum user-owned Base ETH reserve required for Ryu's delegated "
            "source-chain approvals and LI.FI transactions."
        ),
    )
    RYU_SOLANA_GAS_RESERVE_USDC: float = Field(
        default=1.0,
        description=(
            "Amount of a Solana-bound Ryu trade converted into the user's SOL "
            "network-fee reserve when the destination wallet needs gas."
        ),
    )
    RYU_SOLANA_MIN_GAS_RESERVE_SOL: float = Field(
        default=0.002,
        description="Minimum user-owned SOL balance required before Ryu submits an automated transaction.",
    )
    SOLANA_RPC_URL: str = Field(
        default="https://api.mainnet-beta.solana.com",
        description="Solana mainnet RPC used for Ryu balance and gas-reserve checks.",
    )
    YUKI_POSITION_SIZE_MULTIPLIER: float = Field(
        default=1.0,
        description="Legacy compatibility setting. Live Yuki execution preserves the signal's final collateral percentage exactly.",
    )
    YUKI_ENTRY_PRICE_IMPROVEMENT_FRACTION: float = Field(
        default=0.25,
        description="Fraction of the distance from the signal entry toward live price used for Yuki's bounded resting limit.",
    )
    YUKI_ENTRY_ORDER_TTL_HOURS: float = Field(
        default=12.0,
        description="Maximum lifetime of an unfilled Yuki entry; the effective deadline is also capped at half of thesis validity.",
    )
    YUKI_ENTRY_TTL_VALIDITY_FRACTION: float = Field(
        default=0.50,
        description="Maximum fraction of the thesis validity window during which an unfilled Yuki entry may remain live.",
    )
    YUKI_MAX_ENTRY_REVISIONS: int = Field(
        default=2,
        description="Maximum number of fresh entry revisions allowed before an unfilled signal becomes a terminal missed entry.",
    )
    YUKI_FILL_CALIBRATION_LOOKBACK_DAYS: int = Field(
        default=90,
        description="Historical Yuki Hyperliquid order window used to train fill probability.",
    )
    YUKI_FILL_CALIBRATION_MIN_SAMPLES: int = Field(
        default=30,
        description="Minimum terminal Yuki orders required before enabling the trained fill model.",
    )
    YUKI_FILL_CALIBRATION_REFRESH_HOURS: int = Field(
        default=6,
        description="How often the in-memory Hyperliquid fill model retrains from production outcomes.",
    )
    YUKI_PORTFOLIO_SELECTION_ENABLED: bool = Field(
        default=True,
        description="Rank Yuki signals by expected profit per dollar at stop risk before executing them.",
    )
    YUKI_EXECUTION_SERVICE_NAME: str = Field(
        default="platform-signals-worker",
        description="Railway service allowed to submit and monitor autonomous Yuki orders.",
    )
    YUKI_CORRELATED_EXPOSURE_PENALTY: float = Field(
        default=0.35,
        description="Soft utility penalty per existing same-direction exposure in a correlated Yuki risk bucket.",
    )
    YUKI_ENTRY_REVALIDATION_PRICE_MOVE_PCT: float = Field(
        default=0.0125,
        description="Absolute move from the last pending-entry validation that requests an event-driven thesis re-evaluation.",
    )
    YUKI_ENTRY_REVALIDATION_VOLUME_CHANGE_PCT: float = Field(
        default=0.10,
        description="Rolling quote-volume increase from the last pending-entry validation that requests an event-driven re-evaluation.",
    )
    YUKI_ENTRY_REVALIDATION_COOLDOWN_MINUTES: int = Field(
        default=30,
        description="Minimum minutes between event-driven re-evaluation requests for the same pending entry.",
    )
    SIGNAL_MIN_NET_EXPECTED_VALUE_PCT: float = Field(
        default=0.0,
        description="Minimum cost-adjusted unleveraged expected value for calibrated-policy acceptance when sufficient evidence exists.",
    )
    YUKI_TARGET_1_EXIT_FRACTION: float = Field(
        default=0.25,
        description="Fraction of a Yuki position closed at target 1 when target 2 is also present.",
    )
    YUKI_TARGET_2_EXIT_FRACTION: float = Field(
        default=0.25,
        description="Fraction of the original Yuki position closed at target 2; the balance remains as the runner.",
    )
    YUKI_TARGET_CONFIRMATION_BUFFER_PCT: float = Field(
        default=0.0015,
        description="Legacy target confirmation buffer retained for backward-compatible runner safety gaps.",
    )
    YUKI_TARGET_CONFIRMATION_MINUTES: float = Field(
        default=3.0,
        description="Continuous minutes beyond TP1/TP2 that confirm a target milestone.",
    )
    YUKI_TARGET_CONFIRMATION_CANDLES: int = Field(
        default=2,
        description="Consecutive closed one-minute candles beyond TP1/TP2 that confirm a target milestone.",
    )
    YUKI_TARGET_1_STOP_BUFFER_PCT: float = Field(
        default=0.0020,
        description="Minimum price buffer below/above TP1 for the ratcheted runner stop.",
    )
    YUKI_TARGET_1_STOP_ATR_FRACTION: float = Field(
        default=0.25,
        description="ATR fraction used when it is wider than the TP1 percentage buffer.",
    )
    YUKI_TARGET_2_STOP_BUFFER_PCT: float = Field(
        default=0.0015,
        description="Minimum price buffer below/above TP2 before the runner trail starts.",
    )
    YUKI_TARGET_2_STOP_ATR_FRACTION: float = Field(
        default=0.20,
        description="ATR fraction used when it is wider than the TP2 percentage buffer.",
    )
    YUKI_RUNNER_PROFIT_FLOOR_PCT: float = Field(
        default=0.0010,
        description="Minimum profit above/below entry retained by a confirmed-target stop to cover fees and slippage.",
    )
    YUKI_WINNER_SCALE_IN_ENABLED: bool = Field(
        default=True,
        description="Allow one risk-capped add after a platform-signal position becomes a confirmed winner.",
    )
    YUKI_WINNER_SCALE_IN_TRIGGER_R: float = Field(
        default=0.75,
        description="Favorable move, measured in original signal risk units, required before Yuki may add once.",
    )
    YUKI_WINNER_SCALE_IN_MAX_ORIGINAL_SIZE_FRACTION: float = Field(
        default=0.25,
        description="Maximum added size as a fraction of the original platform-signal position.",
    )
    YUKI_WINNER_SCALE_IN_MAX_INITIAL_RISK_FRACTION: float = Field(
        default=0.25,
        description="Maximum loss at the tightened stop contributed by an add, as a fraction of the signal's initial dollar risk.",
    )
    YUKI_RUNNER_TRAIL_ATR_MULTIPLIER: float = Field(
        default=1.75,
        description="ATR distance used by the post-target-2 Yuki runner stop.",
    )
    YUKI_RUNNER_TRAIL_GIVEBACK_FRACTION: float = Field(
        default=0.30,
        description="Maximum fraction of the runner's best price move that the dynamic stop may give back.",
    )
    YUKI_RUNNER_TRAIL_MIN_STEP_PCT: float = Field(
        default=0.0025,
        description="Minimum tighter stop movement before Yuki replaces an exchange trigger, limiting order churn.",
    )
    YUKI_RUNNER_ATR_INTERVAL: str = Field(default="15m")
    YUKI_RUNNER_ATR_PERIOD: int = Field(default=14)
    YUKI_RUNNER_ATR_CACHE_SECONDS: int = Field(default=300)
    YUKI_DEFAULT_HOLD_HOURS: float = Field(
        default=24.0,
        description="Fallback interval before Yuki performs a ReAct position review when a signal has no parseable time horizon.",
    )
    YUKI_MAX_HOLD_HOURS: float = Field(
        default=168.0,
        description="Latest initial ReAct review checkpoint for a Yuki position; it is not a forced-exit cap.",
    )
    YUKI_HORIZON_REVIEW_LOCK_MINUTES: float = Field(
        default=15.0,
        description="Stale-lock window that prevents duplicate ReAct executions across overlapping monitor cycles or restarts; not a review schedule.",
    )
    YUKI_REACT_PRICE_MOVE_TRIGGER_PCT: float = Field(
        default=3.0,
        description="Absolute price move from entry or the last ReAct review that triggers a new position decision.",
    )
    YUKI_REACT_PNL_MOVE_TRIGGER_POINTS: float = Field(
        default=10.0,
        description="Absolute return-on-margin percentage-point change that triggers a new position decision.",
    )
    YUKI_REACT_LLM_PROVIDER: str = Field(
        default="deepseek",
        description="Provider used for Yuki's live ReAct reviews and signal re-entry checks.",
    )
    YUKI_REACT_LLM_MODEL: str = Field(
        default="deepseek-v4-flash",
        description="Lower-cost model used for Yuki's live reviews; final published signal generation keeps the primary model.",
    )
    YUKI_REACT_THINKING_TYPE: str = Field(
        default="disabled",
        description="Thinking mode for Yuki live reviews (enabled or disabled).",
    )
    YUKI_REACT_REASONING_EFFORT: str = Field(
        default="low",
        description="Reasoning effort for Yuki live reviews when the provider supports it.",
    )
    YUKI_REACT_STRICT_JSON: bool = Field(
        default=True,
        description="Force strict JSON output for Yuki live reviews so malformed replies do not trigger retry calls.",
    )
    YUKI_REACT_MAX_OUTPUT_TOKENS: int = Field(
        default=1200,
        description="Output token ceiling for Yuki live reviews and signal re-entry checks.",
    )
    YUKI_REACT_GOALS_LIMIT: int = Field(
        default=3,
        description="Maximum number of active goals included in Yuki live-review prompts.",
    )
    YUKI_REACT_LESSONS_LIMIT: int = Field(
        default=4,
        description="Maximum number of lessons included in Yuki live-review prompts.",
    )
    YUKI_REACT_EPISODES_LIMIT: int = Field(
        default=3,
        description="Maximum number of recent episodes included in Yuki live-review prompts.",
    )
    YUKI_REACT_EVIDENCE_LIST_LIMIT: int = Field(
        default=6,
        description="Maximum list length retained per evidence source in Yuki live-review prompts.",
    )
    YUKI_REACT_EVIDENCE_TEXT_LIMIT: int = Field(
        default=280,
        description="Maximum text length retained per field in Yuki live-review prompts.",
    )
    YUKI_REACT_REVIEW_CACHE_MINUTES: int = Field(
        default=20,
        description="Skip duplicate Yuki live reviews when the event and market-state hash has not materially changed.",
    )
    YUKI_MOVE_STOP_TO_BREAKEVEN_AFTER_TP1: bool = Field(
        default=False,
        description="Move the software stop to entry immediately after target 1; disabled so the target-2 runner can breathe until the profit buffer is reached.",
    )
    YUKI_FUNDING_EXECUTION_ENABLED: bool = Field(
        default=False,
        description="Deprecated recovery-only ETH funding execution. The product flow uses Base USDC with user-paid fees.",
    )
    YUKI_FUNDING_SPONSOR_GAS: bool = Field(
        default=False,
        description="Deprecated and ignored. Floww never sponsors funding gas.",
    )
    HYPERLIQUID_BRIDGE_ADDRESS: str = Field(
        default="0x2Df1c51E09aECF9cacB7bc98cB1742757f163dF7",
        description="Hyperliquid USDC bridge on Arbitrum One (mainnet).",
    )
    HYPERLIQUID_BUILDER_ADDRESS: Optional[str] = Field(
        default=None,
        description="Floww builder-code wallet address for per-order revenue. Unset disables builder fees.",
    )
    HYPERLIQUID_BUILDER_FEE_TENTH_BPS: int = Field(
        default=10,
        description="Builder fee attached to orders, in tenths of a basis point (10 = 1bp = 0.01%).",
    )
    HYPERLIQUID_BUILDER_MAX_FEE_RATE: str = Field(
        default="0.05%",
        description="Maximum builder fee rate the user approves once (caps future per-order fees).",
    )
    
    # Sentiment Analysis Settings
    SENTIMENT_UPDATE_INTERVAL: int = Field(default=1800, description="Sentiment update interval in seconds (30 min)")
    SENTIMENT_CACHE_DURATION: int = Field(default=300, description="Sentiment cache duration in seconds (5 min)")
    SENTIMENT_MIN_CONFIDENCE: float = Field(default=0.3, description="Minimum confidence threshold for sentiment")
    SENTIMENT_WEIGHT_NEWS: float = Field(default=0.25, description="Weight for news sentiment in unified score")
    SENTIMENT_WEIGHT_REDDIT: float = Field(default=0.20, description="Weight for Reddit sentiment in unified score")
    SENTIMENT_WEIGHT_TWITTER: float = Field(default=0.25, description="Weight for Twitter sentiment in unified score")
    SENTIMENT_WEIGHT_MACRO: float = Field(default=0.30, description="Weight for macro sentiment in unified score")
    
    # LLM Configuration - Multiple Providers
    ANTHROPIC_API_KEY: Optional[str] = Field(default=None, description="Claude API key for LLM analysis")
    DEEPSEEK_API_KEY: Optional[str] = Field(default=None, description="DeepSeek API key for LLM analysis")
    OPENAI_API_KEY: Optional[str] = Field(default=None, description="OpenAI API key for LLM analysis")
    LLM_PROVIDER: str = Field(default="deepseek", description="LLM provider (claude/deepseek/openai for trading)")
    # deepseek-reasoner is a legacy alias DeepSeek retires on 2026-07-24;
    # deepseek-v4-pro is the reasoning-tier successor (1M ctx, thinking on by default).
    LLM_MODEL: str = Field(default="deepseek-v4-pro", description="LLM model for trading analysis")
    LLM_TEMPERATURE: float = Field(default=0.1, description="LLM temperature for consistent analysis")
    # NOTE: 1800 was too small for the required output JSON (recommendation, confidence,
    # free-text reasoning, key_factors, entry/targets/stop, risk_factors, risk_assessment),
    # especially with the verbose deepseek-reasoner model. The JSON was truncated →
    # unmatched braces → parse failure → the call returned None ~42% of the time, which
    # read downstream as "AI did not provide a direction" and starved signal generation.
    # The truncation heuristic in unified_signal_generator already assumes ~8K headroom.
    LLM_MAX_TOKENS: int = Field(default=8000, description="Maximum tokens for LLM responses. Must fit the full structured-JSON signal output; 1800 truncated it and caused ~42% None decisions. It is a ceiling (billed on actual output), so generous is safe.")
    PLATFORM_REANALYSIS_COALESCE_MINUTES: int = Field(
        default=30,
        description="Minimum time to wait before re-running the same pending event re-analysis when its state hash has not changed.",
    )
    SIGNAL_GENERATION_EXPENSIVE_TOP_K: int = Field(
        default=10,
        description="Maximum number of top-ranked scheduled candidates to send into the expensive published-signal LLM path per run.",
    )

    # Champion/Challenger configuration (cost-capped verifier)
    CHALLENGER_ENABLED: bool = Field(default=False, description="Enable challenger model verification for selected trades")
    CHALLENGER_PROVIDER: str = Field(default="openai", description="Challenger provider (claude/deepseek/openai)")
    CHALLENGER_MODEL: str = Field(default="gpt-5-mini", description="Challenger model identifier")
    CHALLENGER_MAX_TOKENS: int = Field(default=700, description="Max tokens for challenger verification responses")
    CHALLENGER_TEMPERATURE: float = Field(default=0.0, description="Temperature for challenger verification")
    CHALLENGER_TIMEOUT_SECONDS: int = Field(default=45, description="Timeout for challenger calls")
    CHALLENGER_MAX_SIGNALS_PER_RUN: int = Field(default=2, description="Maximum challenger checks per analysis run")
    CHALLENGER_CONFIDENCE_THRESHOLD: float = Field(default=0.75, description="Minimum confidence required before challenging")
    CHALLENGER_MIN_OPPORTUNITY_SCORE: float = Field(default=0.75, description="Minimum opportunity score required before challenging")
    CHALLENGER_MIN_EXPOSURE_SCORE: float = Field(default=0.8, description="Minimum exposure score (leverage * position fraction) before challenging")
    CHALLENGER_REJECT_ACTION: str = Field(default="skip", description="Action on hard challenger rejection (skip/downsize)")
    CHALLENGER_MONTHLY_BUDGET_USD: float = Field(default=0.0, description="Hard monthly challenger budget cap in USD (0 disables budget enforcement)")
    REJECT_TUNING_ENABLED: bool = Field(default=True, description="Enable confidence gate auto-tuning from rejected-signal outcomes")
    REJECT_TUNING_LOOKBACK_DAYS: int = Field(default=30, description="Lookback window in days for reject-quality tuning")
    REJECT_TUNING_MIN_RESOLVED: int = Field(default=30, description="Minimum resolved rejected candidates before tuning applies")
    REJECT_TUNING_TARGET_BAD_RATE: float = Field(default=0.30, description="Target bad-reject rate used by confidence gate tuner")
    REJECT_TUNING_GAIN: float = Field(default=0.18, description="Sensitivity multiplier for reject-quality confidence adjustment")
    REJECT_TUNING_TARGET_LEVERAGED_PNL_PCT: float = Field(default=-0.20, description="Target average leveraged counterfactual PnL% for rejected candidates (negative means healthy rejects)")
    REJECT_TUNING_PNL_GAIN: float = Field(default=0.012, description="Sensitivity of confidence adjustment to average leveraged reject PnL drift")
    REJECT_TUNING_PNL_WEIGHT: float = Field(default=0.40, description="Blend weight for leveraged PnL term in reject tuning (0-1)")
    REJECT_TUNING_MAX_ABS_ADJUSTMENT: float = Field(default=0.04, description="Maximum absolute confidence adjustment from reject tuning")
    REJECT_TUNING_SMOOTHING: float = Field(default=0.35, description="EMA smoothing factor for reject-quality confidence adjustment")
    REJECT_TUNING_UPDATE_COOLDOWN_HOURS: int = Field(default=6, description="Minimum hours between persisted reject-tuning updates")
    REJECT_COUNTERFACTUAL_MIN_BAD_PNL_PCT: float = Field(
        default=1.0,
        description="Minimum net unleveraged PnL% required before a rejected candidate target hit is treated as a bad reject",
    )
    REJECT_COUNTERFACTUAL_MIN_GOOD_LOSS_PCT: float = Field(
        default=0.5,
        description="Minimum net unleveraged loss magnitude required before a rejected candidate is treated as a good reject",
    )
    REJECT_COUNTERFACTUAL_REPLAY_FRACTION: float = Field(
        default=0.50,
        description="Fraction of validity window to replay before resolving rejected-candidate counterfactuals",
    )
    REJECT_COUNTERFACTUAL_MIN_REPLAY_HOURS: float = Field(
        default=6.0,
        description="Minimum replay window for rejected-candidate counterfactuals before resolution",
    )
    REJECT_COUNTERFACTUAL_MAX_REPLAY_HOURS: float = Field(
        default=24.0,
        description="Maximum replay window for rejected-candidate counterfactuals before resolution",
    )

    # Offline signal policy — loads from disk when enabled; auto-train can refresh the bundle on a schedule
    POLICY_LEARNING_ENABLED: bool = Field(
        default=True,
        description="If true, load policy bundle when present and apply gates + prompt context. Set false to disable policy entirely.",
    )
    POLICY_AUTO_TRAIN_ENABLED: bool = Field(
        default=True,
        description="Retrain policy from Supabase resolved signals on an interval (no manual train_signal_policy.py)",
    )
    POLICY_AUTO_TRAIN_INTERVAL_HOURS: int = Field(
        default=24,
        description="Minimum hours between auto-retrains when a bundle file already exists",
    )
    POLICY_AUTO_TRAIN_RETRY_HOURS_WHEN_MISSING: int = Field(
        default=6,
        description="When no bundle exists yet, minimum hours between training attempts (insufficient data skips save)",
    )
    POLICY_AUTO_TRAIN_LOOKBACK_DAYS: int = Field(default=90, description="Training lookback window for auto-train")
    POLICY_AUTO_TRAIN_MIN_SAMPLES: int = Field(default=60, description="Minimum resolved win/loss rows required to write a bundle")
    POLICY_MODEL_PATH: Optional[str] = Field(default=None, description="Override path to policy_bundle.joblib; default backend/data/signal_policy/policy_bundle.joblib")
    POLICY_PRE_LLM_VETO_THRESHOLD: float = Field(
        default=0.36,
        description="If rule direction P(win) is below this, skip LLM and reject (saves cost on hopeless contexts)",
    )
    POLICY_MIN_WIN_PROB_FOR_ACCEPT: float = Field(
        default=0.42,
        description="Minimum P(win) for final directional acceptance after LLM (guardrail)",
    )
    POLICY_MIN_CONF_ADJUST_SCALE: float = Field(
        default=0.08,
        description="Scale for shifting ensemble min_conf from policy: min_conf -= scale * (p_win - 0.5)",
    )

    # Cost estimation knobs (USD per 1M tokens; set from your billing sheet)
    PRIMARY_LLM_INPUT_COST_PER_1M: float = Field(default=0.0, description="Primary model input cost in USD per 1M tokens")
    PRIMARY_LLM_OUTPUT_COST_PER_1M: float = Field(default=0.0, description="Primary model output cost in USD per 1M tokens")
    CHALLENGER_LLM_INPUT_COST_PER_1M: float = Field(default=0.0, description="Challenger model input cost in USD per 1M tokens")
    CHALLENGER_LLM_OUTPUT_COST_PER_1M: float = Field(default=0.0, description="Challenger model output cost in USD per 1M tokens")
    
    # Helio Configuration
    HELIO_WEBHOOK_SECRET: Optional[str] = Field(default=None, description="Helio webhook secret for payment processing")
    
    # Logging configuration
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    
    # Note: Redis caching removed - using in-memory caching instead
    
    @property
    def cors_origins(self) -> List[str]:
        """Get CORS origins list."""
        origins = [
            # Local development
            "http://localhost:3000",  # Vite dev server (current)
            "http://127.0.0.1:3000",  # Vite dev server alternative (current)
            "http://localhost:5173",  # Vite dev server
            "http://localhost:5175",  # Vite dev server
            "http://localhost:5176",  # Vite dev server
            "http://localhost:5177",  # Vite dev server
            "http://localhost:5178",  # Vite dev server
            "http://127.0.0.1:5173",  # Vite dev server alternative
            "http://127.0.0.1:5175",  # Vite dev server alternative
            "http://localhost:8000",  # Backend API server
            "http://localhost:8001",  # Backend API server
            "http://127.0.0.1:8000",  # Backend API server alternative
            "http://127.0.0.1:8001",  # Backend API server alternative
            
            # Production domains
            "https://tryfloww.app",                  # Custom domain
            "https://www.tryfloww.app",              # Custom domain with www
            
            # Dynamic frontend URL from env
            self.FRONTEND_URL,
        ]

        # Remove duplicates and None values, maintain order
        seen = set()
        unique_origins = []
        for origin in origins:
            if origin and origin not in seen:
                unique_origins.append(origin)
                seen.add(origin)

        return unique_origins

    # Pendle configuration
    PENDLE_API_URL: Optional[str] = Field(default="https://api-v2.pendle.finance/core", description="Pendle API URL")
    PENDLE_CHAIN_ID: Optional[str] = Field(default="8453", description="Pendle Chain ID")

    # Team Agent (Group Trading) configuration
    GROUP_TRADING_TELEGRAM: bool = Field(default=True, description="Enable Team Agent Telegram integration")
    TELEGRAM_BOT_TOKEN: Optional[str] = Field(default=None, description="Telegram bot token for Team Agent integration")

    # Blockchain Configuration
    USE_TESTNET: bool = Field(default=True, description="Use testnet for development")
    # Alchemy API configuration
    ALCHEMY_API_KEY: Optional[str] = Field(default=None, description="Alchemy API key for blockchain data")
    BASE_RPC_URL: str = Field(default="https://base-sepolia.g.alchemy.com/v2/", description="Base network RPC URL")
    CHAIN_ID: int = Field(default=84532, description="Chain ID (84532 for Base Sepolia)")

    # Safe Configuration (updated for testnet)
    SAFE_CORE_SDK_API_URL: str = Field(default="https://safe-transaction-base-sepolia.safe.global", description="Safe Transaction Service API URL")
    SAFE_RELAY_SERVICE_URL: str = Field(default="https://safe-relay.gnosis.io", description="Safe Relay Service URL for transaction sponsoring")

    # Contract Addresses (testnet)
    GOVERNANCE_CONTRACT_ADDRESS: Optional[str] = Field(default=None, description="Deployed governance contract address")
    TESTNET_USDC_ADDRESS: str = Field(default="0x036CbD53842c5426634e7929541eC2318f3dCF7e", description="Base Sepolia USDC address")

    BACKEND_PRIVATE_KEY: Optional[str] = Field(default=None, description="Backend private key for signing transactions")

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "allow"


# Global settings instance
from dotenv import load_dotenv
load_dotenv()  # Force load .env file
settings = Settings()

def get_settings() -> Settings:
    """Get settings instance."""
    return settings
