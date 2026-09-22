-- Virtuals Platform Integration Database Schema
-- This schema supports Flow AI agents integration with Virtuals platform

-- Platform integrations table
CREATE TABLE IF NOT EXISTS platform_integrations (
    id SERIAL PRIMARY KEY,
    platform VARCHAR(50) NOT NULL,
    registration_successful BOOLEAN NOT NULL DEFAULT FALSE,
    registered_agents JSONB DEFAULT '[]',
    network VARCHAR(20) DEFAULT 'base',
    environment VARCHAR(20) DEFAULT 'production',
    api_key_hash VARCHAR(255),
    last_health_check TIMESTAMP,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Virtuals user requests table
CREATE TABLE IF NOT EXISTS virtuals_user_requests (
    id SERIAL PRIMARY KEY,
    request_id VARCHAR(255) UNIQUE NOT NULL,
    user_wallet VARCHAR(42) NOT NULL,
    agent_id VARCHAR(50) NOT NULL,
    request_type VARCHAR(20) NOT NULL, -- 'analyze' or 'execute'
    parameters JSONB NOT NULL DEFAULT '{}',
    payment_amount DECIMAL(18, 8) NOT NULL,
    payment_currency VARCHAR(10) NOT NULL DEFAULT 'ETH',
    status VARCHAR(20) NOT NULL DEFAULT 'pending', -- 'pending', 'processing', 'completed', 'failed'
    result JSONB DEFAULT NULL,
    error_message TEXT DEFAULT NULL,
    transaction_hash VARCHAR(66) DEFAULT NULL,
    estimated_pnl DECIMAL(18, 8) DEFAULT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP DEFAULT NULL
);

-- Virtuals agent performance metrics
CREATE TABLE IF NOT EXISTS virtuals_agent_metrics (
    id SERIAL PRIMARY KEY,
    agent_id VARCHAR(50) NOT NULL,
    date DATE NOT NULL,
    total_requests INTEGER DEFAULT 0,
    successful_requests INTEGER DEFAULT 0,
    failed_requests INTEGER DEFAULT 0,
    total_revenue DECIMAL(18, 8) DEFAULT 0,
    avg_response_time_ms INTEGER DEFAULT 0,
    unique_users INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(agent_id, date)
);

-- Virtuals payment tracking
CREATE TABLE IF NOT EXISTS virtuals_payments (
    id SERIAL PRIMARY KEY,
    request_id VARCHAR(255) NOT NULL,
    user_wallet VARCHAR(42) NOT NULL,
    agent_id VARCHAR(50) NOT NULL,
    amount DECIMAL(18, 8) NOT NULL,
    currency VARCHAR(10) NOT NULL,
    payment_status VARCHAR(20) NOT NULL DEFAULT 'pending', -- 'pending', 'confirmed', 'failed'
    virtuals_payment_id VARCHAR(255),
    transaction_hash VARCHAR(66),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    confirmed_at TIMESTAMP DEFAULT NULL
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_platform_integrations_platform ON platform_integrations(platform);
CREATE INDEX IF NOT EXISTS idx_platform_integrations_timestamp ON platform_integrations(timestamp);

CREATE INDEX IF NOT EXISTS idx_virtuals_requests_user_wallet ON virtuals_user_requests(user_wallet);
CREATE INDEX IF NOT EXISTS idx_virtuals_requests_agent_id ON virtuals_user_requests(agent_id);
CREATE INDEX IF NOT EXISTS idx_virtuals_requests_status ON virtuals_user_requests(status);
CREATE INDEX IF NOT EXISTS idx_virtuals_requests_created_at ON virtuals_user_requests(created_at);

CREATE INDEX IF NOT EXISTS idx_virtuals_metrics_agent_date ON virtuals_agent_metrics(agent_id, date);
CREATE INDEX IF NOT EXISTS idx_virtuals_metrics_date ON virtuals_agent_metrics(date);

CREATE INDEX IF NOT EXISTS idx_virtuals_payments_user_wallet ON virtuals_payments(user_wallet);
CREATE INDEX IF NOT EXISTS idx_virtuals_payments_agent_id ON virtuals_payments(agent_id);
CREATE INDEX IF NOT EXISTS idx_virtuals_payments_status ON virtuals_payments(payment_status);
CREATE INDEX IF NOT EXISTS idx_virtuals_payments_created_at ON virtuals_payments(created_at);

-- Insert initial platform integration record
INSERT INTO platform_integrations (platform, registration_successful, registered_agents, network, environment)
VALUES ('virtuals', FALSE, '[]', 'base', 'production')
ON CONFLICT DO NOTHING;
