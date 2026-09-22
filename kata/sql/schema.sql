-- =====================================================
-- Kata Autonomous Perpetual Agent - Complete Database Schema
-- Consolidated for standalone deployment
-- =====================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";


-- >>> BEGIN FILE: supabase_schema.sql <<<
-- =====================================================
-- Flow AI Trading Platform - Supabase Database Schema
-- =====================================================

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Enable Row Level Security
ALTER DATABASE postgres SET row_security = on;

-- =====================================================
-- 1. USERS TABLE
-- =====================================================
CREATE TABLE users (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    external_wallet VARCHAR(42) UNIQUE NOT NULL,
    platform_wallet VARCHAR(42) UNIQUE,
    selected_agent VARCHAR(20) DEFAULT 'sakura' CHECK (selected_agent IN ('sakura', 'ryu', 'yuki')),
    risk_tolerance INTEGER DEFAULT 5 CHECK (risk_tolerance >= 1 AND risk_tolerance <= 10),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Users table indexes
CREATE INDEX idx_users_external_wallet ON users(external_wallet);
CREATE INDEX idx_users_platform_wallet ON users(platform_wallet);
CREATE INDEX idx_users_selected_agent ON users(selected_agent);
CREATE INDEX idx_users_is_active ON users(is_active);

-- Users table RLS
ALTER TABLE users ENABLE ROW LEVEL SECURITY;

-- Policy: Users can only see and modify their own data
CREATE POLICY "Users can view own data" ON users
    FOR SELECT USING (auth.uid()::text = id::text);

CREATE POLICY "Users can update own data" ON users
    FOR UPDATE USING (auth.uid()::text = id::text);

CREATE POLICY "Users can insert own data" ON users
    FOR INSERT WITH CHECK (auth.uid()::text = id::text);

-- =====================================================
-- 2. AGENT_CONFIGS TABLE
-- =====================================================
CREATE TABLE agent_configs (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    agent_type VARCHAR(20) NOT NULL CHECK (agent_type IN ('sakura', 'ryu', 'yuki')),
    is_active BOOLEAN DEFAULT false,
    max_position_size DECIMAL(5,2) DEFAULT 5.00 CHECK (max_position_size > 0 AND max_position_size <= 100),
    stop_loss_percent DECIMAL(5,2) DEFAULT 3.00 CHECK (stop_loss_percent > 0 AND stop_loss_percent <= 50),
    take_profit_percent DECIMAL(5,2) DEFAULT 15.00 CHECK (take_profit_percent > 0 AND take_profit_percent <= 1000),
    config_data JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Ensure only one active agent per user
    CONSTRAINT unique_active_agent_per_user UNIQUE (user_id, is_active) DEFERRABLE INITIALLY DEFERRED
);

-- Add partial unique index for active agents (only one active agent per user)
CREATE UNIQUE INDEX idx_agent_configs_user_active 
ON agent_configs (user_id) 
WHERE is_active = true;

-- Agent configs table indexes
CREATE INDEX idx_agent_configs_user_id ON agent_configs(user_id);
CREATE INDEX idx_agent_configs_agent_type ON agent_configs(agent_type);
CREATE INDEX idx_agent_configs_is_active ON agent_configs(is_active);

-- Agent configs table RLS
ALTER TABLE agent_configs ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can manage own agent configs" ON agent_configs
    FOR ALL USING (
        EXISTS (
            SELECT 1 FROM users WHERE users.id = agent_configs.user_id AND auth.uid()::text = users.id::text
        )
    );

-- =====================================================
-- 3. PORTFOLIOS TABLE
-- =====================================================
CREATE TABLE portfolios (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    total_value_usd DECIMAL(15,2) DEFAULT 0 CHECK (total_value_usd >= 0),
    total_pnl_usd DECIMAL(15,2) DEFAULT 0,
    total_pnl_percent DECIMAL(8,4) DEFAULT 0,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- One portfolio per user
    CONSTRAINT unique_portfolio_per_user UNIQUE (user_id)
);

-- Portfolios table indexes
CREATE INDEX idx_portfolios_user_id ON portfolios(user_id);
CREATE INDEX idx_portfolios_total_value ON portfolios(total_value_usd);
CREATE INDEX idx_portfolios_last_updated ON portfolios(last_updated);

-- Portfolios table RLS
ALTER TABLE portfolios ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can manage own portfolio" ON portfolios
    FOR ALL USING (
        EXISTS (
            SELECT 1 FROM users WHERE users.id = portfolios.user_id AND auth.uid()::text = users.id::text
        )
    );

-- =====================================================
-- 4. HOLDINGS TABLE
-- =====================================================
CREATE TABLE holdings (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    portfolio_id UUID NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
    token_address VARCHAR(42) NOT NULL,
    token_symbol VARCHAR(20) NOT NULL,
    amount DECIMAL(30,18) NOT NULL CHECK (amount >= 0),
    value_usd DECIMAL(15,2) NOT NULL CHECK (value_usd >= 0),
    pnl_usd DECIMAL(15,2) DEFAULT 0,
    protocol VARCHAR(50) DEFAULT 'base' CHECK (protocol IN ('base', 'pendle', 'lifi', 'hyperliquid', 'uniswap', 'aave')),
    position_type VARCHAR(20) DEFAULT 'spot' CHECK (position_type IN ('spot', 'yield', 'futures', 'options')),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Unique constraint: one holding per token per portfolio per protocol
    CONSTRAINT unique_holding_per_token_protocol UNIQUE (portfolio_id, token_address, protocol, position_type)
);

-- Holdings table indexes
CREATE INDEX idx_holdings_portfolio_id ON holdings(portfolio_id);
CREATE INDEX idx_holdings_token_address ON holdings(token_address);
CREATE INDEX idx_holdings_token_symbol ON holdings(token_symbol);
CREATE INDEX idx_holdings_protocol ON holdings(protocol);
CREATE INDEX idx_holdings_position_type ON holdings(position_type);
CREATE INDEX idx_holdings_value_usd ON holdings(value_usd);
CREATE INDEX idx_holdings_updated_at ON holdings(updated_at);

-- Holdings table RLS
ALTER TABLE holdings ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can manage own holdings" ON holdings
    FOR ALL USING (
        EXISTS (
            SELECT 1 FROM portfolios 
            JOIN users ON users.id = portfolios.user_id 
            WHERE portfolios.id = holdings.portfolio_id 
            AND auth.uid()::text = users.id::text
        )
    );

-- =====================================================
-- 5. TRADES TABLE
-- =====================================================
CREATE TABLE trades (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    agent_type VARCHAR(20) CHECK (agent_type IN ('sakura', 'ryu', 'yuki', 'manual')),
    trade_type VARCHAR(10) NOT NULL CHECK (trade_type IN ('buy', 'sell', 'open', 'close')),
    token_symbol VARCHAR(20) NOT NULL,
    amount DECIMAL(30,18) NOT NULL CHECK (amount > 0),
    price_usd DECIMAL(15,8) NOT NULL CHECK (price_usd > 0),
    value_usd DECIMAL(15,2) NOT NULL CHECK (value_usd > 0),
    protocol VARCHAR(50) DEFAULT 'base' CHECK (protocol IN ('base', 'pendle', 'lifi', 'hyperliquid', 'uniswap', 'aave')),
    tx_hash VARCHAR(66),
    reasoning TEXT,
    confidence_score DECIMAL(3,2) CHECK (confidence_score >= 0 AND confidence_score <= 1),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Trades table indexes
CREATE INDEX idx_trades_user_id ON trades(user_id);
CREATE INDEX idx_trades_agent_type ON trades(agent_type);
CREATE INDEX idx_trades_trade_type ON trades(trade_type);
CREATE INDEX idx_trades_token_symbol ON trades(token_symbol);
CREATE INDEX idx_trades_protocol ON trades(protocol);
CREATE INDEX idx_trades_tx_hash ON trades(tx_hash);
CREATE INDEX idx_trades_created_at ON trades(created_at DESC);
CREATE INDEX idx_trades_value_usd ON trades(value_usd);

-- Trades table RLS
ALTER TABLE trades ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can manage own trades" ON trades
    FOR ALL USING (
        EXISTS (
            SELECT 1 FROM users WHERE users.id = trades.user_id AND auth.uid()::text = users.id::text
        )
    );

-- =====================================================
-- 6. AGENT_DECISIONS TABLE
-- =====================================================
CREATE TABLE agent_decisions (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    agent_type VARCHAR(20) NOT NULL CHECK (agent_type IN ('sakura', 'ryu', 'yuki')),
    decision_type VARCHAR(50) NOT NULL CHECK (decision_type IN (
        'buy_signal', 'sell_signal', 'hold_signal', 'risk_assessment', 
        'portfolio_rebalance', 'stop_loss_trigger', 'take_profit_trigger'
    )),
    market_data JSONB DEFAULT '{}',
    decision_data JSONB DEFAULT '{}',
    action_taken VARCHAR(100),
    confidence_score DECIMAL(3,2) CHECK (confidence_score >= 0 AND confidence_score <= 1),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Agent decisions table indexes
CREATE INDEX idx_agent_decisions_user_id ON agent_decisions(user_id);
CREATE INDEX idx_agent_decisions_agent_type ON agent_decisions(agent_type);
CREATE INDEX idx_agent_decisions_decision_type ON agent_decisions(decision_type);
CREATE INDEX idx_agent_decisions_created_at ON agent_decisions(created_at DESC);
CREATE INDEX idx_agent_decisions_confidence_score ON agent_decisions(confidence_score);

-- GIN index for JSONB columns
CREATE INDEX idx_agent_decisions_market_data ON agent_decisions USING GIN (market_data);
CREATE INDEX idx_agent_decisions_decision_data ON agent_decisions USING GIN (decision_data);

-- Agent decisions table RLS
ALTER TABLE agent_decisions ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can manage own agent decisions" ON agent_decisions
    FOR ALL USING (
        EXISTS (
            SELECT 1 FROM users WHERE users.id = agent_decisions.user_id AND auth.uid()::text = users.id::text
        )
    );

-- =====================================================
-- 7. PERFORMANCE_METRICS TABLE
-- =====================================================
CREATE TABLE performance_metrics (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    agent_type VARCHAR(20) CHECK (agent_type IN ('sakura', 'ryu', 'yuki', 'overall')),
    metric_date DATE NOT NULL DEFAULT CURRENT_DATE,
    portfolio_value_usd DECIMAL(15,2) NOT NULL CHECK (portfolio_value_usd >= 0),
    daily_pnl_usd DECIMAL(15,2) DEFAULT 0,
    trades_count INTEGER DEFAULT 0 CHECK (trades_count >= 0),
    win_rate DECIMAL(5,2) DEFAULT 0 CHECK (win_rate >= 0 AND win_rate <= 100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Unique constraint: one metric per user per agent per date
    CONSTRAINT unique_metric_per_user_agent_date UNIQUE (user_id, agent_type, metric_date)
);

-- Performance metrics table indexes
CREATE INDEX idx_performance_metrics_user_id ON performance_metrics(user_id);
CREATE INDEX idx_performance_metrics_agent_type ON performance_metrics(agent_type);
CREATE INDEX idx_performance_metrics_metric_date ON performance_metrics(metric_date DESC);
CREATE INDEX idx_performance_metrics_portfolio_value ON performance_metrics(portfolio_value_usd);
CREATE INDEX idx_performance_metrics_win_rate ON performance_metrics(win_rate);

-- Performance metrics table RLS
ALTER TABLE performance_metrics ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can manage own performance metrics" ON performance_metrics
    FOR ALL USING (
        EXISTS (
            SELECT 1 FROM users WHERE users.id = performance_metrics.user_id AND auth.uid()::text = users.id::text
        )
    );

-- =====================================================
-- FUNCTIONS AND TRIGGERS
-- =====================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply updated_at triggers
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_agent_configs_updated_at BEFORE UPDATE ON agent_configs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_holdings_updated_at BEFORE UPDATE ON holdings
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to automatically create portfolio for new users
CREATE OR REPLACE FUNCTION create_user_portfolio()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO portfolios (user_id) VALUES (NEW.id);
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger to create portfolio for new users
CREATE TRIGGER create_portfolio_for_new_user AFTER INSERT ON users
    FOR EACH ROW EXECUTE FUNCTION create_user_portfolio();

-- Function to update portfolio total value when holdings change
CREATE OR REPLACE FUNCTION update_portfolio_value()
RETURNS TRIGGER AS $$
DECLARE
    portfolio_total DECIMAL(15,2);
    portfolio_pnl DECIMAL(15,2);
BEGIN
    -- Calculate new totals
    SELECT 
        COALESCE(SUM(value_usd), 0),
        COALESCE(SUM(pnl_usd), 0)
    INTO portfolio_total, portfolio_pnl
    FROM holdings 
    WHERE portfolio_id = COALESCE(NEW.portfolio_id, OLD.portfolio_id);
    
    -- Update portfolio
    UPDATE portfolios 
    SET 
        total_value_usd = portfolio_total,
        total_pnl_usd = portfolio_pnl,
        total_pnl_percent = CASE 
            WHEN portfolio_total > 0 THEN (portfolio_pnl / portfolio_total) * 100
            ELSE 0
        END,
        last_updated = NOW()
    WHERE id = COALESCE(NEW.portfolio_id, OLD.portfolio_id);
    
    RETURN COALESCE(NEW, OLD);
END;
$$ language 'plpgsql';

-- Triggers to update portfolio value
CREATE TRIGGER update_portfolio_on_holding_change 
    AFTER INSERT OR UPDATE OR DELETE ON holdings
    FOR EACH ROW EXECUTE FUNCTION update_portfolio_value();

-- =====================================================
-- VIEWS FOR COMMON QUERIES
-- =====================================================

-- View: User portfolio summary
CREATE VIEW user_portfolio_summary AS
SELECT 
    u.id as user_id,
    u.external_wallet,
    u.platform_wallet,
    u.selected_agent,
    p.total_value_usd,
    p.total_pnl_usd,
    p.total_pnl_percent,
    p.last_updated,
    COUNT(h.id) as total_holdings,
    ac.agent_type as active_agent_type,
    ac.is_active as agent_is_active
FROM users u
LEFT JOIN portfolios p ON u.id = p.user_id
LEFT JOIN holdings h ON p.id = h.portfolio_id
LEFT JOIN agent_configs ac ON u.id = ac.user_id AND ac.is_active = true
GROUP BY u.id, p.id, ac.id;

-- View: Agent performance summary
CREATE VIEW agent_performance_summary AS
SELECT 
    user_id,
    agent_type,
    COUNT(*) as total_trades,
    SUM(CASE WHEN trade_type IN ('buy', 'open') THEN value_usd ELSE 0 END) as total_buy_volume,
    SUM(CASE WHEN trade_type IN ('sell', 'close') THEN value_usd ELSE 0 END) as total_sell_volume,
    AVG(confidence_score) as avg_confidence,
    DATE_TRUNC('day', created_at) as trade_date
FROM trades 
WHERE agent_type IS NOT NULL
GROUP BY user_id, agent_type, DATE_TRUNC('day', created_at);

-- =====================================================
-- SAMPLE DATA INSERTION (Optional)
-- =====================================================

-- Insert sample agent configurations
INSERT INTO users (id, external_wallet, platform_wallet) VALUES 
    ('11111111-1111-1111-1111-111111111111', '0x1234567890123456789012345678901234567890', '0x0987654321098765432109876543210987654321');

-- Grant necessary permissions for RLS to work with auth
GRANT USAGE ON SCHEMA public TO anon, authenticated;
GRANT ALL ON ALL TABLES IN SCHEMA public TO authenticated;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO authenticated;

-- =====================================================
-- COMMENTS FOR DOCUMENTATION
-- =====================================================

COMMENT ON TABLE users IS 'Core user accounts with wallet addresses and preferences';
COMMENT ON TABLE agent_configs IS 'AI agent configurations and trading parameters per user';
COMMENT ON TABLE portfolios IS 'Portfolio summaries and total values per user';
COMMENT ON TABLE holdings IS 'Individual token holdings across different protocols';
COMMENT ON TABLE trades IS 'Trade history with agent decisions and transaction details';
COMMENT ON TABLE agent_decisions IS 'AI agent decision logs with market analysis';
COMMENT ON TABLE performance_metrics IS 'Daily performance tracking for agents and portfolios';

COMMENT ON COLUMN users.risk_tolerance IS 'Risk tolerance scale 1-10 (1=very conservative, 10=very aggressive)';
COMMENT ON COLUMN agent_configs.max_position_size IS 'Maximum position size as percentage of portfolio';
COMMENT ON COLUMN trades.confidence_score IS 'AI confidence in trade decision (0.0-1.0)';
COMMENT ON COLUMN performance_metrics.win_rate IS 'Percentage of profitable trades (0-100)';
-- >>> END FILE: supabase_schema.sql <<<

-- >>> BEGIN FILE: create_platform_signals_tables.sql <<<
-- Platform Signals Database Schema
-- Creates tables for shared platform signal architecture

-- Platform Signals Table (shared signals for all users)
CREATE TABLE IF NOT EXISTS platform_signals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    signal_id VARCHAR(255) UNIQUE NOT NULL,
    token_symbol VARCHAR(50) NOT NULL,
    direction VARCHAR(10) NOT NULL CHECK (direction IN ('LONG', 'SHORT')),
    timeframe VARCHAR(10) NOT NULL,
    
    -- Signal Details
    confidence DECIMAL(5,4) NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    overall_score DECIMAL(5,4) NOT NULL CHECK (overall_score >= 0 AND overall_score <= 1),
    signal_strength VARCHAR(20) NOT NULL,
    time_horizon VARCHAR(20) NOT NULL,
    
    -- Price Targets
    entry_price DECIMAL(20,8) NOT NULL,
    target_1 DECIMAL(20,8) NOT NULL,
    target_1_probability DECIMAL(5,4) NOT NULL,
    target_2 DECIMAL(20,8) NOT NULL,
    target_2_probability DECIMAL(5,4) NOT NULL,
    stop_loss DECIMAL(20,8) NOT NULL,
    risk_reward_ratio DECIMAL(8,4) NOT NULL,
    
    -- Market Analysis
    market_conditions JSONB DEFAULT '{}',
    technical_indicators JSONB DEFAULT '{}',
    sentiment_data JSONB DEFAULT '{}',
    risk_factors TEXT[] DEFAULT '{}',
    
    -- AI Analysis Data
    ai_reasoning TEXT,
    ai_key_factors TEXT[],
    ai_risk_assessment TEXT,
    ai_confidence_breakdown JSONB DEFAULT '{}',
    
    -- Platform Signal Metadata
    signal_pool VARCHAR(20) NOT NULL CHECK (signal_pool IN ('fast_mode', 'full_mode', 'premium')),
    opportunity_rank INTEGER NOT NULL,
    analysis_timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    validity_window_hours INTEGER NOT NULL DEFAULT 72,
    
    -- Status Tracking
    status VARCHAR(20) NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'expired', 'invalidated')),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    -- Notes and Analysis
    analysis_notes TEXT,
    generation_source VARCHAR(50) NOT NULL DEFAULT 'platform_analysis',

    -- Analysis Run Tracking (for duplicate prevention)
    run_id VARCHAR(255)
);


-- User Platform Signal Tracking (tracks user interaction with platform signals)
CREATE TABLE IF NOT EXISTS user_platform_signal_tracking (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    platform_signal_id UUID NOT NULL REFERENCES platform_signals(id) ON DELETE CASCADE,
    signal_id VARCHAR(255) NOT NULL, -- Denormalized for faster queries
    
    -- User Interaction
    viewed_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    followed BOOLEAN NOT NULL DEFAULT FALSE,
    followed_at TIMESTAMP WITH TIME ZONE,
    
    -- User-Specific Trading Data
    user_entry_price DECIMAL(20,8),
    user_exit_price DECIMAL(20,8),
    user_exit_timestamp TIMESTAMP WITH TIME ZONE,
    user_position_size DECIMAL(20,8),
    
    -- User P&L Tracking
    profit_loss_percentage DECIMAL(8,4),
    profit_loss_absolute DECIMAL(20,8),
    user_outcome VARCHAR(20) CHECK (user_outcome IN ('win', 'loss', 'scratch', 'active', 'expired')),
    
    -- Metadata
    delivery_mode VARCHAR(20) NOT NULL CHECK (delivery_mode IN ('fast_mode', 'full_mode')),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    UNIQUE(user_id, platform_signal_id)
);

-- NOTE: Token Universe and Analysis Queue tables removed
-- Platform signals now use full mode scanner for dynamic token discovery
-- No pre-configured token lists needed - discovers opportunities automatically

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_platform_signals_symbol ON platform_signals(token_symbol);
CREATE INDEX IF NOT EXISTS idx_platform_signals_pool ON platform_signals(signal_pool);
CREATE INDEX IF NOT EXISTS idx_platform_signals_rank ON platform_signals(opportunity_rank);
CREATE INDEX IF NOT EXISTS idx_platform_signals_status ON platform_signals(status);
CREATE INDEX IF NOT EXISTS idx_platform_signals_expires ON platform_signals(expires_at);
CREATE INDEX IF NOT EXISTS idx_platform_signals_created ON platform_signals(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_platform_signals_run_id ON platform_signals(run_id);

-- Add unique constraint to prevent duplicate signals within same analysis run
-- This ensures no duplicate symbol+direction+timeframe combinations per run
CREATE UNIQUE INDEX IF NOT EXISTS idx_platform_signals_unique_per_run
ON platform_signals(token_symbol, direction, timeframe, run_id)
WHERE status = 'active' AND run_id IS NOT NULL;


CREATE INDEX IF NOT EXISTS idx_user_tracking_user_id ON user_platform_signal_tracking(user_id);
CREATE INDEX IF NOT EXISTS idx_user_tracking_signal_id ON user_platform_signal_tracking(platform_signal_id);
CREATE INDEX IF NOT EXISTS idx_user_tracking_followed ON user_platform_signal_tracking(followed) WHERE followed = TRUE;

-- Indexes for removed token_universe and analysis_queue tables no longer needed

-- Add triggers for updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_platform_signals_updated_at BEFORE UPDATE ON platform_signals FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_user_platform_signal_tracking_updated_at BEFORE UPDATE ON user_platform_signal_tracking FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Initial token seeding removed - platform now uses dynamic token discovery
-- Full mode scanner automatically discovers and analyzes top market opportunities

-- Add comments for documentation
COMMENT ON TABLE platform_signals IS 'Shared trading signals generated by platform analysis using full mode scanner';
COMMENT ON TABLE user_platform_signal_tracking IS 'Individual user interaction and P&L tracking for platform signals';
-- >>> END FILE: create_platform_signals_tables.sql <<<

-- >>> BEGIN FILE: add_platform_signal_performance_tracking.sql <<<
-- Create platform signal performance tracking table for historical max profits
-- This table tracks the maximum profit/loss reached by each signal throughout its lifetime
-- Essential for proper performance metrics that account for all targets (TARGET_1, TARGET_2, etc.)

CREATE TABLE IF NOT EXISTS platform_signal_performance_tracking (
    id SERIAL PRIMARY KEY,
    signal_id VARCHAR(255) UNIQUE NOT NULL,
    max_profit_reached DECIMAL(10, 4) DEFAULT 0.0000,
    max_loss_reached DECIMAL(10, 4) DEFAULT 0.0000,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign key to platform_signals table
    FOREIGN KEY (signal_id) REFERENCES platform_signals(signal_id) ON DELETE CASCADE
);

-- Index for performance
CREATE INDEX IF NOT EXISTS idx_platform_signal_performance_signal_id ON platform_signal_performance_tracking(signal_id);
CREATE INDEX IF NOT EXISTS idx_platform_signal_performance_updated ON platform_signal_performance_tracking(last_updated);

-- Add comments for clarity
COMMENT ON TABLE platform_signal_performance_tracking IS 'Tracks historical maximum profit/loss reached by platform signals across all targets';
COMMENT ON COLUMN platform_signal_performance_tracking.signal_id IS 'Unique signal identifier from platform_signals table';
COMMENT ON COLUMN platform_signal_performance_tracking.max_profit_reached IS 'Maximum profit percentage reached during signal lifetime (positive values only)';
COMMENT ON COLUMN platform_signal_performance_tracking.max_loss_reached IS 'Maximum loss percentage reached during signal lifetime (negative values only)';
COMMENT ON COLUMN platform_signal_performance_tracking.last_updated IS 'Last time the max profit/loss values were updated';
-- >>> END FILE: add_platform_signal_performance_tracking.sql <<<

-- >>> BEGIN FILE: create_yuki_agent_tables.sql <<<
-- Yuki Agent Database Tables
-- Creates all necessary tables for Yuki agent functionality

-- Platform Wallet Balances Table
CREATE TABLE IF NOT EXISTS platform_wallet_balances (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    wallet_address VARCHAR(255) NOT NULL,

    -- Balance tracking
    total_balance DECIMAL(20,8) NOT NULL DEFAULT 0.00000000,
    allocated_balance DECIMAL(20,8) NOT NULL DEFAULT 0.00000000,
    available_balance DECIMAL(20,8) NOT NULL DEFAULT 0.00000000,

    -- Token breakdown
    usdc_balance DECIMAL(20,8) NOT NULL DEFAULT 0.00000000,
    eth_balance DECIMAL(20,8) NOT NULL DEFAULT 0.00000000,
    cbeth_balance DECIMAL(20,8) NOT NULL DEFAULT 0.00000000,
    steth_balance DECIMAL(20,8) NOT NULL DEFAULT 0.00000000,

    -- Metadata
    last_sync_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    UNIQUE(user_id, wallet_address)
);

-- Agent Configurations Table
CREATE TABLE IF NOT EXISTS agent_configurations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    agent_type VARCHAR(50) NOT NULL CHECK (agent_type IN ('yuki', 'sakura', 'ryu')),

    -- Configuration
    risk_tolerance VARCHAR(20) NOT NULL CHECK (risk_tolerance IN ('low', 'medium', 'high', 'extreme')),
    max_position_size DECIMAL(5,2) NOT NULL DEFAULT 10.00, -- Percentage
    max_leverage INTEGER NOT NULL DEFAULT 5,
    stop_loss_percentage DECIMAL(5,2) NOT NULL DEFAULT 5.00,
    take_profit_percentage DECIMAL(5,2) NOT NULL DEFAULT 15.00,

    -- Agent-specific settings (JSON for flexibility)
    agent_settings JSONB DEFAULT '{}',

    -- Status
    is_active BOOLEAN NOT NULL DEFAULT FALSE,
    is_delegated BOOLEAN NOT NULL DEFAULT FALSE,

    -- Performance tracking
    total_trades INTEGER NOT NULL DEFAULT 0,
    winning_trades INTEGER NOT NULL DEFAULT 0,
    total_pnl DECIMAL(20,8) NOT NULL DEFAULT 0.00000000,
    max_drawdown DECIMAL(5,2) NOT NULL DEFAULT 0.00,

    -- Metadata
    activated_at TIMESTAMP WITH TIME ZONE,
    deactivated_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    UNIQUE(user_id, agent_type)
);

-- User Trades Table
CREATE TABLE IF NOT EXISTS user_trades (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    agent_type VARCHAR(50) NOT NULL,

    -- Trade details
    token_symbol VARCHAR(50) NOT NULL,
    side VARCHAR(10) NOT NULL CHECK (side IN ('LONG', 'SHORT')),
    position_size DECIMAL(20,8) NOT NULL,
    leverage INTEGER NOT NULL DEFAULT 1,

    -- Prices
    entry_price DECIMAL(20,8) NOT NULL,
    exit_price DECIMAL(20,8),
    stop_loss DECIMAL(20,8),
    take_profit DECIMAL(20,8),

    -- P&L
    pnl_usd DECIMAL(20,8) DEFAULT 0.00000000,
    pnl_percentage DECIMAL(8,4) DEFAULT 0.0000,
    fees_paid DECIMAL(20,8) DEFAULT 0.00000000,

    -- Status and timing
    status VARCHAR(20) NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'closed', 'liquidated', 'cancelled')),
    opened_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    closed_at TIMESTAMP WITH TIME ZONE,

    -- External references
    hyperliquid_order_id VARCHAR(255),
    signal_id VARCHAR(255), -- Reference to platform signal

    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Platform Wallet Strategies Table
CREATE TABLE IF NOT EXISTS platform_wallet_strategies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    agent_name VARCHAR(50) NOT NULL,
    strategy_type VARCHAR(50) NOT NULL,

    -- Allocation
    allocated_amount DECIMAL(20,8) NOT NULL,
    allocated_percentage DECIMAL(5,2) NOT NULL,

    -- Performance
    current_value DECIMAL(20,8) NOT NULL DEFAULT 0.00000000,
    total_pnl DECIMAL(20,8) NOT NULL DEFAULT 0.00000000,
    daily_pnl DECIMAL(20,8) NOT NULL DEFAULT 0.00000000,

    -- Status
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    auto_rebalance BOOLEAN NOT NULL DEFAULT FALSE,

    -- Strategy settings
    strategy_config JSONB DEFAULT '{}',

    -- Metadata
    last_rebalance_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Agent Performance Metrics Table
CREATE TABLE IF NOT EXISTS agent_performance_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    agent_type VARCHAR(50) NOT NULL,

    -- Time period
    period_type VARCHAR(20) NOT NULL CHECK (period_type IN ('daily', 'weekly', 'monthly')),
    period_start TIMESTAMP WITH TIME ZONE NOT NULL,
    period_end TIMESTAMP WITH TIME ZONE NOT NULL,

    -- Performance metrics
    total_trades INTEGER NOT NULL DEFAULT 0,
    winning_trades INTEGER NOT NULL DEFAULT 0,
    losing_trades INTEGER NOT NULL DEFAULT 0,
    win_rate DECIMAL(5,2) NOT NULL DEFAULT 0.00,

    -- P&L metrics
    gross_pnl DECIMAL(20,8) NOT NULL DEFAULT 0.00000000,
    net_pnl DECIMAL(20,8) NOT NULL DEFAULT 0.00000000,
    total_fees DECIMAL(20,8) NOT NULL DEFAULT 0.00000000,

    -- Risk metrics
    max_drawdown DECIMAL(5,2) NOT NULL DEFAULT 0.00,
    sharpe_ratio DECIMAL(8,4) DEFAULT 0.0000,
    volatility DECIMAL(5,2) DEFAULT 0.00,

    -- Volume metrics
    total_volume DECIMAL(20,8) NOT NULL DEFAULT 0.00000000,
    avg_position_size DECIMAL(20,8) NOT NULL DEFAULT 0.00000000,

    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    UNIQUE(user_id, agent_type, period_type, period_start)
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_platform_wallet_balances_user_id ON platform_wallet_balances(user_id);
CREATE INDEX IF NOT EXISTS idx_platform_wallet_balances_wallet_address ON platform_wallet_balances(wallet_address);

CREATE INDEX IF NOT EXISTS idx_agent_configurations_user_id ON agent_configurations(user_id);
CREATE INDEX IF NOT EXISTS idx_agent_configurations_agent_type ON agent_configurations(agent_type);
CREATE INDEX IF NOT EXISTS idx_agent_configurations_active ON agent_configurations(is_active) WHERE is_active = TRUE;

CREATE INDEX IF NOT EXISTS idx_user_trades_user_id ON user_trades(user_id);
CREATE INDEX IF NOT EXISTS idx_user_trades_agent_type ON user_trades(agent_type);
CREATE INDEX IF NOT EXISTS idx_user_trades_status ON user_trades(status);
CREATE INDEX IF NOT EXISTS idx_user_trades_opened_at ON user_trades(opened_at DESC);

CREATE INDEX IF NOT EXISTS idx_platform_wallet_strategies_user_id ON platform_wallet_strategies(user_id);
CREATE INDEX IF NOT EXISTS idx_platform_wallet_strategies_agent_name ON platform_wallet_strategies(agent_name);
CREATE INDEX IF NOT EXISTS idx_platform_wallet_strategies_active ON platform_wallet_strategies(is_active) WHERE is_active = TRUE;

CREATE INDEX IF NOT EXISTS idx_agent_performance_metrics_user_agent ON agent_performance_metrics(user_id, agent_type);
CREATE INDEX IF NOT EXISTS idx_agent_performance_metrics_period ON agent_performance_metrics(period_type, period_start DESC);

-- Add triggers for updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_platform_wallet_balances_updated_at BEFORE UPDATE ON platform_wallet_balances FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_agent_configurations_updated_at BEFORE UPDATE ON agent_configurations FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_user_trades_updated_at BEFORE UPDATE ON user_trades FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_platform_wallet_strategies_updated_at BEFORE UPDATE ON platform_wallet_strategies FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_agent_performance_metrics_updated_at BEFORE UPDATE ON agent_performance_metrics FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Add comments for documentation
COMMENT ON TABLE platform_wallet_balances IS 'User wallet balances and token allocations for agent trading';
COMMENT ON TABLE agent_configurations IS 'Individual user agent configurations and settings';
COMMENT ON TABLE user_trades IS 'Individual trade records executed by agents for users';
COMMENT ON TABLE platform_wallet_strategies IS 'User allocation strategies across different agents';
COMMENT ON TABLE agent_performance_metrics IS 'Historical performance metrics for agents by user';
-- >>> END FILE: create_yuki_agent_tables.sql <<<

-- >>> BEGIN FILE: add_yuki_execution_tables.sql <<<
-- Yuki allocation and execution tables
-- Supports Hyperliquid-backed allocation intents, trade execution records, and live position polling.


CREATE TABLE IF NOT EXISTS agent_allocations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    allocation_id TEXT NOT NULL UNIQUE,
    user_id TEXT NOT NULL,
    agent_type TEXT NOT NULL CHECK (agent_type IN ('yuki', 'sakura', 'ryu')),
    allocated_amount NUMERIC(20, 8) NOT NULL,
    remaining_amount NUMERIC(20, 8) NOT NULL,
    status TEXT NOT NULL DEFAULT 'paused' CHECK (status IN ('active', 'paused', 'stopped', 'pending')),
    platform_wallet_address TEXT,
    trading_config JSONB NOT NULL DEFAULT '{}',
    performance_metrics JSONB NOT NULL DEFAULT '{}',
    last_trade_at TIMESTAMPTZ,
    total_trades INTEGER NOT NULL DEFAULT 0,
    realized_pnl NUMERIC(20, 8) NOT NULL DEFAULT 0,
    unrealized_pnl NUMERIC(20, 8) NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS agent_trade_executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    execution_id TEXT NOT NULL UNIQUE,
    allocation_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    agent_type TEXT NOT NULL CHECK (agent_type IN ('yuki', 'sakura', 'ryu')),
    symbol TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('buy', 'sell')),
    size NUMERIC(30, 12) NOT NULL,
    price NUMERIC(30, 12) NOT NULL,
    amount_used NUMERIC(20, 8) NOT NULL,
    leverage NUMERIC(8, 3),
    executed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    hyperliquid_order_id TEXT,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'filled', 'failed', 'cancelled')),
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS agent_trades (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    allocation_id TEXT NOT NULL,
    agent_type TEXT NOT NULL CHECK (agent_type IN ('yuki', 'sakura', 'ryu')),
    trade_type TEXT NOT NULL CHECK (trade_type IN ('open_long', 'open_short', 'close_long', 'close_short', 'close')),
    symbol TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('buy', 'sell')),
    entry_price NUMERIC(30, 12) NOT NULL,
    exit_price NUMERIC(30, 12),
    position_size NUMERIC(30, 12) NOT NULL,
    leverage NUMERIC(8, 3) NOT NULL DEFAULT 1,
    trade_amount NUMERIC(20, 8) NOT NULL,
    realized_pnl NUMERIC(20, 8),
    unrealized_pnl NUMERIC(20, 8),
    fees NUMERIC(20, 8) DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'filled', 'partially_filled', 'cancelled', 'failed', 'closed')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    filled_at TIMESTAMPTZ,
    closed_at TIMESTAMPTZ,
    hyperliquid_order_id TEXT,
    tx_hash TEXT,
    signal_confidence NUMERIC(8, 6),
    signal_reasoning TEXT,
    error_message TEXT,
    trade_metadata JSONB DEFAULT '{}',
    max_profit_reached NUMERIC(20, 8),
    max_loss_reached NUMERIC(20, 8),
    time_in_position_minutes INTEGER
);

CREATE TABLE IF NOT EXISTS agent_positions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    allocation_id TEXT NOT NULL,
    trade_id UUID REFERENCES agent_trades(id) ON DELETE SET NULL,
    agent_type TEXT NOT NULL CHECK (agent_type IN ('yuki', 'sakura', 'ryu')),
    symbol TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('long', 'short')),
    size NUMERIC(30, 12) NOT NULL,
    entry_price NUMERIC(30, 12) NOT NULL,
    current_price NUMERIC(30, 12) NOT NULL,
    leverage NUMERIC(8, 3) NOT NULL,
    position_value NUMERIC(20, 8) NOT NULL,
    unrealized_pnl NUMERIC(20, 8) NOT NULL DEFAULT 0,
    unrealized_pnl_percent NUMERIC(12, 6) NOT NULL DEFAULT 0,
    margin_used NUMERIC(20, 8) NOT NULL,
    liquidation_price NUMERIC(30, 12),
    margin_ratio NUMERIC(12, 8),
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    hyperliquid_position_id TEXT,
    position_metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_agent_allocations_user_id ON agent_allocations(user_id);
CREATE INDEX IF NOT EXISTS idx_agent_allocations_status ON agent_allocations(status);
CREATE INDEX IF NOT EXISTS idx_agent_allocations_created_at ON agent_allocations(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_agent_trade_executions_allocation_id ON agent_trade_executions(allocation_id);
CREATE INDEX IF NOT EXISTS idx_agent_trade_executions_user_id ON agent_trade_executions(user_id);
CREATE INDEX IF NOT EXISTS idx_agent_trade_executions_executed_at ON agent_trade_executions(executed_at DESC);

CREATE INDEX IF NOT EXISTS idx_agent_trades_allocation_id ON agent_trades(allocation_id);
CREATE INDEX IF NOT EXISTS idx_agent_trades_user_id ON agent_trades(user_id);
CREATE INDEX IF NOT EXISTS idx_agent_trades_symbol ON agent_trades(symbol);
CREATE INDEX IF NOT EXISTS idx_agent_trades_status ON agent_trades(status);
CREATE INDEX IF NOT EXISTS idx_agent_trades_created_at ON agent_trades(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_agent_positions_allocation_id ON agent_positions(allocation_id);
CREATE INDEX IF NOT EXISTS idx_agent_positions_user_id ON agent_positions(user_id);
CREATE INDEX IF NOT EXISTS idx_agent_positions_active ON agent_positions(is_active) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_agent_positions_symbol ON agent_positions(symbol);

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS update_agent_allocations_updated_at ON agent_allocations;
CREATE TRIGGER update_agent_allocations_updated_at
BEFORE UPDATE ON agent_allocations
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_agent_positions_updated_at ON agent_positions;
CREATE TRIGGER update_agent_positions_updated_at
BEFORE UPDATE ON agent_positions
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- >>> END FILE: add_yuki_execution_tables.sql <<<

-- >>> BEGIN FILE: create_agent_learning_tables.sql <<<
-- Agent Learning System Database Tables
-- Creates tables for storing trade outcomes and learning patterns for AI agents

-- 1. Agent Learning Memory Table
-- Stores detailed information about each trade outcome for learning purposes
CREATE TABLE IF NOT EXISTS agent_learning_memory (
    id SERIAL PRIMARY KEY,
    agent_type VARCHAR(50) NOT NULL, -- 'yuki', 'sakura', 'ryu'
    signal_id VARCHAR(255) NOT NULL,
    token_symbol VARCHAR(20) NOT NULL,
    direction VARCHAR(10) NOT NULL, -- 'LONG', 'SHORT'

    -- Market Context at Trade Time
    market_conditions JSONB NOT NULL,
    technical_indicators JSONB NOT NULL,
    sentiment_data JSONB,

    -- Signal Properties
    signal_confidence DECIMAL(5,4) NOT NULL,
    leverage_used INTEGER NOT NULL DEFAULT 1,
    entry_price DECIMAL(20,8) NOT NULL,
    target_1 DECIMAL(20,8),
    target_2 DECIMAL(20,8),
    stop_loss DECIMAL(20,8),

    -- Trade Outcome
    outcome VARCHAR(20) NOT NULL, -- 'win', 'loss', 'scratch'
    exit_reason VARCHAR(50), -- 'TARGET_1', 'TARGET_2', 'STOP_LOSS', 'EXPIRED'
    pnl_percentage DECIMAL(10,4) NOT NULL,
    max_profit_reached DECIMAL(10,4) DEFAULT 0,
    max_loss_reached DECIMAL(10,4) DEFAULT 0,

    -- Learning Features
    pattern_features JSONB NOT NULL, -- Extracted features for ML
    lesson_learned TEXT,

    -- Metadata
    analysis_timestamp TIMESTAMP NOT NULL,
    exit_timestamp TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Foreign key constraints
    FOREIGN KEY (signal_id) REFERENCES platform_signals(signal_id) ON DELETE CASCADE
);

-- 2. Agent Learning Patterns Table
-- Stores recognized patterns and their success rates for confidence adjustment
CREATE TABLE IF NOT EXISTS agent_learning_patterns (
    id SERIAL PRIMARY KEY,
    agent_type VARCHAR(50) NOT NULL,
    pattern_name VARCHAR(200) NOT NULL,
    pattern_hash VARCHAR(64) UNIQUE NOT NULL, -- Hash of pattern conditions for uniqueness

    -- Pattern Statistics
    success_rate DECIMAL(5,4) NOT NULL DEFAULT 0, -- 0.0 to 1.0
    total_trades INTEGER NOT NULL DEFAULT 0,
    winning_trades INTEGER NOT NULL DEFAULT 0,
    losing_trades INTEGER NOT NULL DEFAULT 0,
    avg_pnl DECIMAL(10,4) NOT NULL DEFAULT 0,
    best_pnl DECIMAL(10,4) DEFAULT 0,
    worst_pnl DECIMAL(10,4) DEFAULT 0,

    -- Pattern Conditions
    conditions JSONB NOT NULL, -- The market/technical conditions that define this pattern
    confidence_boost DECIMAL(5,4) DEFAULT 0, -- How much to boost/reduce confidence (-0.5 to +0.5)

    -- Risk Adjustments
    suggested_leverage_multiplier DECIMAL(3,2) DEFAULT 1.0, -- Multiplier for leverage based on pattern
    suggested_position_size_multiplier DECIMAL(3,2) DEFAULT 1.0,

    -- Metadata
    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_trade_date TIMESTAMP,

    -- Constraints
    CONSTRAINT chk_success_rate CHECK (success_rate >= 0 AND success_rate <= 1),
    CONSTRAINT chk_confidence_boost CHECK (confidence_boost >= -0.5 AND confidence_boost <= 0.5),
    CONSTRAINT chk_leverage_multiplier CHECK (suggested_leverage_multiplier >= 0.1 AND suggested_leverage_multiplier <= 2.0)
);

-- 3. Agent Learning Performance Tracking
-- Tracks how well the learning system is improving agent performance over time
CREATE TABLE IF NOT EXISTS agent_learning_performance (
    id SERIAL PRIMARY KEY,
    agent_type VARCHAR(50) NOT NULL,
    measurement_date DATE NOT NULL DEFAULT CURRENT_DATE,

    -- Performance Metrics
    total_signals INTEGER DEFAULT 0,
    signals_with_learning INTEGER DEFAULT 0, -- Signals where learning was applied

    -- Pre-learning performance (baseline)
    baseline_win_rate DECIMAL(5,4) DEFAULT 0,
    baseline_avg_pnl DECIMAL(10,4) DEFAULT 0,
    baseline_profit_factor DECIMAL(10,4) DEFAULT 0,

    -- Post-learning performance
    learning_win_rate DECIMAL(5,4) DEFAULT 0,
    learning_avg_pnl DECIMAL(10,4) DEFAULT 0,
    learning_profit_factor DECIMAL(10,4) DEFAULT 0,

    -- Improvement metrics
    win_rate_improvement DECIMAL(5,4) DEFAULT 0,
    pnl_improvement DECIMAL(10,4) DEFAULT 0,

    -- Learning system health
    total_patterns_recognized INTEGER DEFAULT 0,
    active_patterns INTEGER DEFAULT 0,
    confidence_adjustments_made INTEGER DEFAULT 0,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Unique constraint
    UNIQUE(agent_type, measurement_date)
);

-- Create indexes separately for PostgreSQL compatibility
CREATE INDEX IF NOT EXISTS idx_agent_learning_memory_agent_type ON agent_learning_memory(agent_type);
CREATE INDEX IF NOT EXISTS idx_agent_learning_memory_token ON agent_learning_memory(token_symbol);
CREATE INDEX IF NOT EXISTS idx_agent_learning_memory_outcome ON agent_learning_memory(outcome);
CREATE INDEX IF NOT EXISTS idx_agent_learning_memory_created ON agent_learning_memory(created_at);
CREATE INDEX IF NOT EXISTS idx_agent_learning_memory_features ON agent_learning_memory USING GIN (pattern_features);

CREATE INDEX IF NOT EXISTS idx_agent_patterns_type ON agent_learning_patterns(agent_type);
CREATE INDEX IF NOT EXISTS idx_agent_patterns_success_rate ON agent_learning_patterns(success_rate);
CREATE INDEX IF NOT EXISTS idx_agent_patterns_updated ON agent_learning_patterns(last_updated);
CREATE INDEX IF NOT EXISTS idx_agent_patterns_conditions ON agent_learning_patterns USING GIN (conditions);

CREATE INDEX IF NOT EXISTS idx_learning_performance_agent ON agent_learning_performance(agent_type);
CREATE INDEX IF NOT EXISTS idx_learning_performance_date ON agent_learning_performance(measurement_date);

-- 4. Helper Functions

-- Function to calculate pattern hash for uniqueness
CREATE OR REPLACE FUNCTION calculate_pattern_hash(conditions JSONB)
RETURNS VARCHAR(64) AS $$
BEGIN
    RETURN encode(sha256(conditions::text::bytea), 'hex');
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Function to extract key pattern features for similarity matching
CREATE OR REPLACE FUNCTION extract_pattern_key(pattern_features JSONB)
RETURNS TEXT AS $$
DECLARE
    key_parts TEXT[];
BEGIN
    -- Extract and combine key features into a pattern key
    key_parts := ARRAY[
        COALESCE(pattern_features->>'market_regime', 'unknown'),
        COALESCE(pattern_features->>'setup_type', 'unknown'),
        COALESCE(pattern_features->>'timeframe_bucket', 'unknown'),
        COALESCE(pattern_features->>'volatility_level', 'unknown'),
        COALESCE(pattern_features->>'rsi_range', 'unknown'),
        COALESCE(pattern_features->>'trend_strength', 'unknown'),
        COALESCE(pattern_features->>'direction', 'unknown'),
        COALESCE(pattern_features->>'time_category', 'unknown')
    ];

    RETURN array_to_string(key_parts, '|');
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Trigger to automatically update pattern statistics when new learning memory is added
CREATE OR REPLACE FUNCTION update_pattern_statistics()
RETURNS TRIGGER AS $$
DECLARE
    pattern_key TEXT;
    pattern_hash_val VARCHAR(64);
BEGIN
    -- Extract pattern key and calculate hash
    pattern_key := extract_pattern_key(NEW.pattern_features);
    pattern_hash_val := calculate_pattern_hash(NEW.pattern_features);

    -- Update or insert pattern statistics
    INSERT INTO agent_learning_patterns (
        agent_type,
        pattern_name,
        pattern_hash,
        conditions,
        total_trades,
        winning_trades,
        losing_trades,
        avg_pnl,
        best_pnl,
        worst_pnl,
        success_rate,
        last_trade_date,
        last_updated
    )
    VALUES (
        NEW.agent_type,
        pattern_key,
        pattern_hash_val,
        NEW.pattern_features,
        1,
        CASE WHEN NEW.outcome = 'win' THEN 1 ELSE 0 END,
        CASE WHEN NEW.outcome = 'loss' THEN 1 ELSE 0 END,
        NEW.pnl_percentage,
        GREATEST(NEW.pnl_percentage, 0),
        LEAST(NEW.pnl_percentage, 0),
        CASE WHEN NEW.outcome = 'win' THEN 1.0 ELSE 0.0 END,
        NEW.analysis_timestamp,
        CURRENT_TIMESTAMP
    )
    ON CONFLICT (pattern_hash) DO UPDATE SET
        total_trades = agent_learning_patterns.total_trades + 1,
        winning_trades = agent_learning_patterns.winning_trades +
            CASE WHEN NEW.outcome = 'win' THEN 1 ELSE 0 END,
        losing_trades = agent_learning_patterns.losing_trades +
            CASE WHEN NEW.outcome = 'loss' THEN 1 ELSE 0 END,
        avg_pnl = (agent_learning_patterns.avg_pnl * agent_learning_patterns.total_trades + NEW.pnl_percentage) /
            (agent_learning_patterns.total_trades + 1),
        best_pnl = GREATEST(agent_learning_patterns.best_pnl, NEW.pnl_percentage),
        worst_pnl = LEAST(agent_learning_patterns.worst_pnl, NEW.pnl_percentage),
        success_rate = (agent_learning_patterns.winning_trades +
            CASE WHEN NEW.outcome = 'win' THEN 1 ELSE 0 END)::DECIMAL /
            (agent_learning_patterns.total_trades + 1),
        confidence_boost = CASE
            WHEN (agent_learning_patterns.winning_trades +
                CASE WHEN NEW.outcome = 'win' THEN 1 ELSE 0 END)::DECIMAL /
                (agent_learning_patterns.total_trades + 1) > 0.65
            THEN LEAST(0.2, ((agent_learning_patterns.winning_trades +
                CASE WHEN NEW.outcome = 'win' THEN 1 ELSE 0 END)::DECIMAL /
                (agent_learning_patterns.total_trades + 1) - 0.5) * 0.4)
            WHEN (agent_learning_patterns.winning_trades +
                CASE WHEN NEW.outcome = 'win' THEN 1 ELSE 0 END)::DECIMAL /
                (agent_learning_patterns.total_trades + 1) < 0.35
            THEN GREATEST(-0.2, ((agent_learning_patterns.winning_trades +
                CASE WHEN NEW.outcome = 'win' THEN 1 ELSE 0 END)::DECIMAL /
                (agent_learning_patterns.total_trades + 1) - 0.5) * 0.4)
            ELSE 0
        END,
        last_trade_date = NEW.analysis_timestamp,
        last_updated = CURRENT_TIMESTAMP;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for automatic pattern updates
DROP TRIGGER IF EXISTS trg_update_pattern_statistics ON agent_learning_memory;
CREATE TRIGGER trg_update_pattern_statistics
    AFTER INSERT ON agent_learning_memory
    FOR EACH ROW
    EXECUTE FUNCTION update_pattern_statistics();

-- Grant permissions
GRANT SELECT, INSERT, UPDATE ON agent_learning_memory TO authenticated, service_role;
GRANT SELECT, INSERT, UPDATE ON agent_learning_patterns TO authenticated, service_role;
GRANT SELECT, INSERT, UPDATE ON agent_learning_performance TO authenticated, service_role;
GRANT EXECUTE ON FUNCTION calculate_pattern_hash(JSONB) TO authenticated, service_role;
GRANT EXECUTE ON FUNCTION extract_pattern_key(JSONB) TO authenticated, service_role;
GRANT EXECUTE ON FUNCTION update_pattern_statistics() TO service_role;

-- Add comments for documentation
COMMENT ON TABLE agent_learning_memory IS 'Stores detailed trade outcomes for agent learning and pattern recognition';
COMMENT ON TABLE agent_learning_patterns IS 'Recognized trading patterns with success rates for confidence adjustment';
COMMENT ON TABLE agent_learning_performance IS 'Tracks learning system performance and improvement over time';
COMMENT ON COLUMN agent_learning_patterns.confidence_boost IS 'Confidence adjustment: positive for good patterns, negative for poor patterns';
COMMENT ON COLUMN agent_learning_patterns.pattern_hash IS 'SHA256 hash of pattern conditions for uniqueness';

-- >>> END FILE: create_agent_learning_tables.sql <<<

-- >>> BEGIN FILE: add_multi_level_learning_tables.sql <<<
-- =====================================================
-- Multi-Level Learning Architecture Database Schema
-- =====================================================
-- This migration adds all necessary tables to support the
-- 3-level learning architecture for trading agents

-- =====================================================
-- 1. AGENT LEARNING INSIGHTS TABLE
-- =====================================================
-- Stores learning outcomes and insights from all three levels
CREATE TABLE IF NOT EXISTS agent_learning_insights (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Agent and Signal Identification
    agent_type VARCHAR(20) NOT NULL CHECK (agent_type IN ('yuki', 'ryu', 'sakura')),
    signal_id VARCHAR(255),

    -- Learning Level and Type
    learning_level VARCHAR(20) NOT NULL CHECK (learning_level IN ('real_time', 'pattern', 'meta')),
    learning_type VARCHAR(50) NOT NULL,

    -- Insight Data
    insights JSONB NOT NULL DEFAULT '{}',
    confidence_score DECIMAL(5,4) CHECK (confidence_score >= 0 AND confidence_score <= 1),

    -- Performance Metrics
    performance_impact JSONB DEFAULT '{}',
    risk_metrics JSONB DEFAULT '{}',

    -- Metadata
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- =====================================================
-- 2. AGENT LEARNING ADJUSTMENTS TABLE
-- =====================================================
-- Tracks real-time adjustment suggestions and their outcomes
CREATE TABLE IF NOT EXISTS agent_learning_adjustments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Signal and Agent Identification
    signal_id VARCHAR(255) NOT NULL,
    agent_type VARCHAR(20) NOT NULL CHECK (agent_type IN ('yuki', 'ryu', 'sakura')),

    -- Adjustment Details
    adjustment_type VARCHAR(50) NOT NULL,
    adjustments JSONB NOT NULL DEFAULT '{}',
    confidence DECIMAL(5,4) NOT NULL CHECK (confidence >= 0 AND confidence <= 1),

    -- Outcome Tracking
    applied BOOLEAN DEFAULT FALSE,
    applied_at TIMESTAMP WITH TIME ZONE,
    outcome_pnl DECIMAL(10,6),
    effectiveness_score DECIMAL(5,4),

    -- Metadata
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- =====================================================
-- 3. MARKET REGIME DETECTION TABLE
-- =====================================================
-- Records market regime changes and characteristics
CREATE TABLE IF NOT EXISTS market_regime_detection (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Market Identification
    token_symbol VARCHAR(50),
    timeframe VARCHAR(10),

    -- Regime Classification
    regime_type VARCHAR(50) NOT NULL,
    strength DECIMAL(5,4) NOT NULL CHECK (strength >= 0 AND strength <= 1),
    confidence DECIMAL(5,4) NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    duration_minutes INTEGER,

    -- Supporting Data
    supporting_factors TEXT[] DEFAULT '{}',
    market_metrics JSONB DEFAULT '{}',
    technical_indicators JSONB DEFAULT '{}',

    -- Detection Metadata
    detected_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    detected_by VARCHAR(50) NOT NULL DEFAULT 'multi_level_learning',

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- =====================================================
-- 4. PROMPT OPTIMIZATION HISTORY TABLE
-- =====================================================
-- Logs AI-driven prompt improvements and their performance
CREATE TABLE IF NOT EXISTS prompt_optimization_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Agent and Context
    agent_type VARCHAR(20) NOT NULL CHECK (agent_type IN ('yuki', 'ryu', 'sakura')),
    prompt_type VARCHAR(50) NOT NULL,
    optimization_version INTEGER NOT NULL DEFAULT 1,

    -- Prompt Content
    original_prompt TEXT NOT NULL,
    optimized_prompt TEXT NOT NULL,
    optimization_rationale TEXT,

    -- Performance Analysis
    success_patterns TEXT[] DEFAULT '{}',
    failure_patterns TEXT[] DEFAULT '{}',
    improvement_suggestions TEXT[] DEFAULT '{}',

    -- Performance Metrics Before/After
    baseline_performance JSONB DEFAULT '{}',
    optimized_performance JSONB DEFAULT '{}',
    performance_delta JSONB DEFAULT '{}',

    -- Metadata
    optimization_timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- =====================================================
-- 5. REAL TIME SIGNAL TRACKING TABLE
-- =====================================================
-- Tracks real-time price movements and feedback for active signals
CREATE TABLE IF NOT EXISTS real_time_signal_tracking (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Signal Identification
    signal_id VARCHAR(255) NOT NULL,
    platform_signal_id UUID REFERENCES platform_signals(id) ON DELETE CASCADE,
    agent_type VARCHAR(20) NOT NULL CHECK (agent_type IN ('yuki', 'ryu', 'sakura')),

    -- Real-Time Data
    current_price DECIMAL(20,8),
    current_pnl DECIMAL(10,6),
    max_favorable_excursion DECIMAL(10,6),
    max_adverse_excursion DECIMAL(10,6),
    time_elapsed_minutes INTEGER,

    -- Market Conditions
    regime_change_detected BOOLEAN DEFAULT FALSE,
    volatility_spike BOOLEAN DEFAULT FALSE,
    volume_confirmation BOOLEAN DEFAULT FALSE,
    support_resistance_test BOOLEAN DEFAULT FALSE,

    -- Feedback and Adjustments
    feedback_confidence DECIMAL(5,4) CHECK (feedback_confidence >= 0 AND feedback_confidence <= 1),
    adjustment_suggested JSONB DEFAULT '{}',
    price_history JSONB DEFAULT '[]',
    pnl_history JSONB DEFAULT '[]',

    -- Status
    tracking_status VARCHAR(20) NOT NULL DEFAULT 'active' CHECK (tracking_status IN ('active', 'completed', 'stopped')),

    -- Timestamps
    started_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    last_update TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- =====================================================
-- 6. PATTERN PERFORMANCE TRACKING TABLE
-- =====================================================
-- Enhanced pattern performance with advanced metrics
CREATE TABLE IF NOT EXISTS pattern_performance_tracking (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Pattern Identification
    pattern_name VARCHAR(100) NOT NULL,
    agent_type VARCHAR(20) NOT NULL CHECK (agent_type IN ('yuki', 'ryu', 'sakura')),

    -- Pattern Features
    pattern_features JSONB NOT NULL DEFAULT '{}',
    market_conditions JSONB DEFAULT '{}',

    -- Performance Metrics
    sample_size INTEGER NOT NULL DEFAULT 0,
    success_rate DECIMAL(5,4) CHECK (success_rate >= 0 AND success_rate <= 1),
    avg_pnl DECIMAL(10,6),
    avg_time_to_target DECIMAL(8,2),

    -- Risk Metrics
    risk_adjusted_return DECIMAL(8,4),
    sharpe_ratio DECIMAL(8,4),
    sortino_ratio DECIMAL(8,4),
    max_drawdown DECIMAL(8,4),
    volatility DECIMAL(8,4),

    -- Correlations
    confidence_correlation DECIMAL(5,4),
    volume_correlation DECIMAL(5,4),

    -- Best Conditions
    best_market_conditions JSONB DEFAULT '{}',
    optimal_timeframes TEXT[] DEFAULT '{}',

    -- Metadata
    last_updated TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    UNIQUE(pattern_name, agent_type)
);

-- =====================================================
-- 7. META LEARNING PERFORMANCE TABLE
-- =====================================================
-- Cross-agent performance comparison and meta insights
CREATE TABLE IF NOT EXISTS meta_learning_performance (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Time Period
    analysis_date DATE NOT NULL,
    agent_type VARCHAR(20) NOT NULL CHECK (agent_type IN ('yuki', 'ryu', 'sakura', 'all')),
    time_period_hours INTEGER NOT NULL DEFAULT 24,

    -- Strategy Effectiveness (per agent)
    strategy_effectiveness JSONB NOT NULL DEFAULT '{}',
    cross_agent_performance JSONB DEFAULT '{}',
    optimal_market_conditions JSONB DEFAULT '{}',

    -- Learning Metrics
    learning_convergence_rate DECIMAL(8,4),
    adaptation_speed_minutes DECIMAL(8,2),
    prompt_optimization_impact DECIMAL(8,4),

    -- Performance Improvements
    win_rate_improvement DECIMAL(8,4),
    sharpe_improvement DECIMAL(8,4),
    drawdown_reduction DECIMAL(8,4),

    -- Recommendations
    adaptation_recommendations TEXT[] DEFAULT '{}',
    risk_parameter_adjustments JSONB DEFAULT '{}',
    temporal_performance_patterns JSONB DEFAULT '{}',

    -- Metadata
    analysis_timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    UNIQUE(analysis_date, agent_type) DEFERRABLE INITIALLY DEFERRED
);

-- Backward compatibility: add column on older deployments where it is missing.
ALTER TABLE meta_learning_performance
ADD COLUMN IF NOT EXISTS agent_type VARCHAR(20) CHECK (agent_type IN ('yuki', 'ryu', 'sakura', 'all'));

-- =====================================================
-- CREATE INDEXES FOR PERFORMANCE
-- =====================================================

-- Agent Learning Insights Indexes
CREATE INDEX IF NOT EXISTS idx_learning_insights_agent_type ON agent_learning_insights(agent_type);
CREATE INDEX IF NOT EXISTS idx_learning_insights_learning_level ON agent_learning_insights(learning_level);
CREATE INDEX IF NOT EXISTS idx_learning_insights_signal_id ON agent_learning_insights(signal_id);
CREATE INDEX IF NOT EXISTS idx_learning_insights_timestamp ON agent_learning_insights(timestamp DESC);

-- Agent Learning Adjustments Indexes
CREATE INDEX IF NOT EXISTS idx_learning_adjustments_signal_id ON agent_learning_adjustments(signal_id);
CREATE INDEX IF NOT EXISTS idx_learning_adjustments_agent_type ON agent_learning_adjustments(agent_type);
CREATE INDEX IF NOT EXISTS idx_learning_adjustments_applied ON agent_learning_adjustments(applied);
CREATE INDEX IF NOT EXISTS idx_learning_adjustments_timestamp ON agent_learning_adjustments(timestamp DESC);

-- Market Regime Detection Indexes
CREATE INDEX IF NOT EXISTS idx_regime_detection_symbol ON market_regime_detection(token_symbol);
CREATE INDEX IF NOT EXISTS idx_regime_detection_regime_type ON market_regime_detection(regime_type);
CREATE INDEX IF NOT EXISTS idx_regime_detection_detected_at ON market_regime_detection(detected_at DESC);

-- Prompt Optimization History Indexes
CREATE INDEX IF NOT EXISTS idx_prompt_optimization_agent_type ON prompt_optimization_history(agent_type);
CREATE INDEX IF NOT EXISTS idx_prompt_optimization_prompt_type ON prompt_optimization_history(prompt_type);
CREATE INDEX IF NOT EXISTS idx_prompt_optimization_version ON prompt_optimization_history(optimization_version DESC);
CREATE INDEX IF NOT EXISTS idx_prompt_optimization_timestamp ON prompt_optimization_history(optimization_timestamp DESC);

-- Real Time Signal Tracking Indexes
CREATE INDEX IF NOT EXISTS idx_realtime_tracking_signal_id ON real_time_signal_tracking(signal_id);
CREATE INDEX IF NOT EXISTS idx_realtime_tracking_agent_type ON real_time_signal_tracking(agent_type);
CREATE INDEX IF NOT EXISTS idx_realtime_tracking_status ON real_time_signal_tracking(tracking_status);
CREATE INDEX IF NOT EXISTS idx_realtime_tracking_last_update ON real_time_signal_tracking(last_update DESC);

-- Pattern Performance Tracking Indexes
CREATE INDEX IF NOT EXISTS idx_pattern_performance_agent_type ON pattern_performance_tracking(agent_type);
CREATE INDEX IF NOT EXISTS idx_pattern_performance_pattern_name ON pattern_performance_tracking(pattern_name);
CREATE INDEX IF NOT EXISTS idx_pattern_performance_success_rate ON pattern_performance_tracking(success_rate DESC);
CREATE INDEX IF NOT EXISTS idx_pattern_performance_last_updated ON pattern_performance_tracking(last_updated DESC);

-- Meta Learning Performance Indexes
CREATE INDEX IF NOT EXISTS idx_meta_learning_analysis_date ON meta_learning_performance(analysis_date DESC);
CREATE INDEX IF NOT EXISTS idx_meta_learning_agent_type ON meta_learning_performance(agent_type);
CREATE INDEX IF NOT EXISTS idx_meta_learning_convergence ON meta_learning_performance(learning_convergence_rate DESC);

-- =====================================================
-- CREATE TRIGGERS FOR UPDATED_AT TIMESTAMPS
-- =====================================================

CREATE TRIGGER update_agent_learning_insights_updated_at
    BEFORE UPDATE ON agent_learning_insights
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_agent_learning_adjustments_updated_at
    BEFORE UPDATE ON agent_learning_adjustments
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_market_regime_detection_updated_at
    BEFORE UPDATE ON market_regime_detection
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_prompt_optimization_history_updated_at
    BEFORE UPDATE ON prompt_optimization_history
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_real_time_signal_tracking_updated_at
    BEFORE UPDATE ON real_time_signal_tracking
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_pattern_performance_tracking_updated_at
    BEFORE UPDATE ON pattern_performance_tracking
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_meta_learning_performance_updated_at
    BEFORE UPDATE ON meta_learning_performance
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =====================================================
-- ADD COMMENTS FOR DOCUMENTATION
-- =====================================================

COMMENT ON TABLE agent_learning_insights IS 'Stores learning outcomes and insights from all three levels of the multi-level learning architecture';
COMMENT ON TABLE agent_learning_adjustments IS 'Tracks real-time adjustment suggestions and their effectiveness outcomes';
COMMENT ON TABLE market_regime_detection IS 'Records market regime changes detected by the learning system';
COMMENT ON TABLE prompt_optimization_history IS 'Logs AI-driven prompt improvements and their performance impact';
COMMENT ON TABLE real_time_signal_tracking IS 'Real-time price and performance tracking for active signals';
COMMENT ON TABLE pattern_performance_tracking IS 'Enhanced pattern performance metrics with advanced risk calculations';
COMMENT ON TABLE meta_learning_performance IS 'Cross-agent performance comparison and meta-learning insights';

-- =====================================================
-- SUCCESS MESSAGE
-- =====================================================
-- This completes the database schema for the Multi-Level Learning Architecture
-- The system can now track:
-- - Real-time signal performance and adjustments
-- - Pattern recognition and performance optimization
-- - Meta-learning insights and cross-agent analysis
-- - Market regime detection and adaptation
-- - AI-driven prompt optimization

-- >>> END FILE: add_multi_level_learning_tables.sql <<<
