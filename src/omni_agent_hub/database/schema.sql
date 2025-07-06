-- Omni-Agent Hub Database Schema
-- PostgreSQL schema for storing agent data, sessions, and analytics

-- Sessions table
CREATE TABLE IF NOT EXISTS sessions (
    session_id VARCHAR(255) PRIMARY KEY,
    user_id VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    context JSONB,
    status VARCHAR(50) DEFAULT 'active',
    metadata JSONB
);

-- Agent memory table
CREATE TABLE IF NOT EXISTS agent_memory (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    agent_name VARCHAR(255) NOT NULL,
    memory_key VARCHAR(255) NOT NULL,
    memory_value TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(session_id, agent_name, memory_key)
);

-- Agent interactions table
CREATE TABLE IF NOT EXISTS agent_interactions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    agent_name VARCHAR(255) NOT NULL,
    user_input TEXT,
    agent_response TEXT,
    success BOOLEAN,
    confidence FLOAT,
    execution_time_ms INTEGER,
    tools_used JSONB,
    react_steps JSONB,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Performance metrics table
CREATE TABLE IF NOT EXISTS performance_metrics (
    id SERIAL PRIMARY KEY,
    metric_type VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    context JSONB,
    tags JSONB,
    source VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Learning patterns table
CREATE TABLE IF NOT EXISTS learning_patterns (
    id SERIAL PRIMARY KEY,
    pattern_id VARCHAR(255) UNIQUE NOT NULL,
    pattern_type VARCHAR(100),
    input_features JSONB,
    output_features JSONB,
    success_rate FLOAT,
    usage_count INTEGER DEFAULT 1,
    confidence_score FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User preferences table
CREATE TABLE IF NOT EXISTS user_preferences (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) UNIQUE NOT NULL,
    preferences JSONB,
    interaction_count INTEGER DEFAULT 0,
    avg_satisfaction FLOAT DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Prompt performance table
CREATE TABLE IF NOT EXISTS prompt_performance (
    id SERIAL PRIMARY KEY,
    prompt_id VARCHAR(255) NOT NULL,
    usage_count INTEGER DEFAULT 0,
    success_rate FLOAT DEFAULT 0.0,
    avg_response_time FLOAT DEFAULT 0.0,
    user_satisfaction FLOAT DEFAULT 0.0,
    optimization_score FLOAT DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tool performance table
CREATE TABLE IF NOT EXISTS tool_performance (
    id SERIAL PRIMARY KEY,
    tool_name VARCHAR(255) NOT NULL,
    usage_count INTEGER DEFAULT 0,
    success_rate FLOAT DEFAULT 0.0,
    avg_execution_time FLOAT DEFAULT 0.0,
    error_rate FLOAT DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Knowledge base table
CREATE TABLE IF NOT EXISTS knowledge_base (
    id SERIAL PRIMARY KEY,
    title VARCHAR(500),
    content TEXT,
    category VARCHAR(100),
    tags JSONB,
    embedding_id VARCHAR(255), -- Reference to Milvus vector
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- System events table
CREATE TABLE IF NOT EXISTS system_events (
    id SERIAL PRIMARY KEY,
    event_type VARCHAR(100) NOT NULL,
    event_data JSONB,
    severity VARCHAR(50) DEFAULT 'info',
    source VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON sessions(created_at);
CREATE INDEX IF NOT EXISTS idx_agent_memory_session ON agent_memory(session_id);
CREATE INDEX IF NOT EXISTS idx_agent_interactions_session ON agent_interactions(session_id);
CREATE INDEX IF NOT EXISTS idx_agent_interactions_created_at ON agent_interactions(created_at);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_type ON performance_metrics(metric_type);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_created_at ON performance_metrics(created_at);
CREATE INDEX IF NOT EXISTS idx_learning_patterns_type ON learning_patterns(pattern_type);
CREATE INDEX IF NOT EXISTS idx_knowledge_base_category ON knowledge_base(category);
CREATE INDEX IF NOT EXISTS idx_system_events_type ON system_events(event_type);
CREATE INDEX IF NOT EXISTS idx_system_events_created_at ON system_events(created_at);

-- Create triggers for updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_sessions_updated_at BEFORE UPDATE ON sessions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_agent_memory_updated_at BEFORE UPDATE ON agent_memory
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_learning_patterns_updated_at BEFORE UPDATE ON learning_patterns
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_preferences_updated_at BEFORE UPDATE ON user_preferences
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_prompt_performance_updated_at BEFORE UPDATE ON prompt_performance
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_tool_performance_updated_at BEFORE UPDATE ON tool_performance
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_knowledge_base_updated_at BEFORE UPDATE ON knowledge_base
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
