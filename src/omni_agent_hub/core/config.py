"""
Configuration management for Omni-Agent Hub.

This module handles all configuration settings using Pydantic Settings
for type safety and validation.
"""

import os
from functools import lru_cache
from typing import List, Optional

from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Application Settings
    app_name: str = Field(default="Omni-Agent Hub", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    
    # Database Configuration
    database_url: str = Field(
        default="postgresql+asyncpg://omni_user:omni_pass@localhost:5432/omni_hub",
        env="DATABASE_URL"
    )
    database_pool_size: int = Field(default=20, env="DATABASE_POOL_SIZE")
    database_max_overflow: int = Field(default=30, env="DATABASE_MAX_OVERFLOW")
    
    # Redis Configuration
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    redis_max_connections: int = Field(default=20, env="REDIS_MAX_CONNECTIONS")
    
    # Milvus Vector Database
    milvus_host: str = Field(default="localhost", env="MILVUS_HOST")
    milvus_port: int = Field(default=19530, env="MILVUS_PORT")
    milvus_user: Optional[str] = Field(default=None, env="MILVUS_USER")
    milvus_password: Optional[str] = Field(default=None, env="MILVUS_PASSWORD")
    milvus_collection_name: str = Field(default="omni_embeddings", env="MILVUS_COLLECTION_NAME")
    
    # Kafka Configuration
    kafka_bootstrap_servers: str = Field(default="localhost:9092", env="KAFKA_BOOTSTRAP_SERVERS")
    kafka_topic_prefix: str = Field(default="omni_hub", env="KAFKA_TOPIC_PREFIX")

    # MinIO Configuration
    minio_endpoint: str = Field(default="localhost:9000", env="MINIO_ENDPOINT")
    minio_access_key: str = Field(default="minioadmin", env="MINIO_ACCESS_KEY")
    minio_secret_key: str = Field(default="minioadmin", env="MINIO_SECRET_KEY")
    minio_secure: bool = Field(default=False, env="MINIO_SECURE")
    
    # AI Model Configuration (2025 Latest)
    # OpenAI Models
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4.1", env="OPENAI_MODEL")
    openai_fallback_model: str = Field(default="gpt-4o", env="OPENAI_FALLBACK_MODEL")
    openai_embedding_model: str = Field(default="text-embedding-3-small", env="OPENAI_EMBEDDING_MODEL")
    openai_vision_model: str = Field(default="gpt-4o", env="OPENAI_VISION_MODEL")
    openai_code_model: str = Field(default="gpt-4.1", env="OPENAI_CODE_MODEL")

    # Anthropic Models
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-3-7-sonnet-latest", env="ANTHROPIC_MODEL")
    anthropic_fallback_model: str = Field(default="claude-3-5-sonnet-20241022", env="ANTHROPIC_FALLBACK_MODEL")

    # Google Gemini Models
    google_api_key: str = Field(default="", env="GOOGLE_API_KEY")
    gemini_model: str = Field(default="gemini-2.5-pro", env="GEMINI_MODEL")

    # Model Selection Strategy
    primary_llm_provider: str = Field(default="openai", env="PRIMARY_LLM_PROVIDER")
    fallback_llm_provider: str = Field(default="anthropic", env="FALLBACK_LLM_PROVIDER")
    auto_fallback_enabled: bool = Field(default=True, env="AUTO_FALLBACK_ENABLED")
    
    # MCP Configuration
    mcp_server_host: str = Field(default="localhost", env="MCP_SERVER_HOST")
    mcp_server_port: int = Field(default=8001, env="MCP_SERVER_PORT")
    mcp_tools_config_path: str = Field(default="./config/mcp_tools.json", env="MCP_TOOLS_CONFIG_PATH")
    
    # Tool Integration APIs
    kagi_api_key: Optional[str] = Field(default=None, env="KAGI_API_KEY")
    aws_access_key_id: Optional[str] = Field(default=None, env="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(default=None, env="AWS_SECRET_ACCESS_KEY")
    aws_region: str = Field(default="us-east-1", env="AWS_REGION")
    
    slack_bot_token: Optional[str] = Field(default=None, env="SLACK_BOT_TOKEN")
    slack_signing_secret: Optional[str] = Field(default=None, env="SLACK_SIGNING_SECRET")
    
    dbt_cloud_api_token: Optional[str] = Field(default=None, env="DBT_CLOUD_API_TOKEN")
    dbt_cloud_account_id: Optional[str] = Field(default=None, env="DBT_CLOUD_ACCOUNT_ID")
    
    # Security Configuration
    secret_key: str = Field(default="change-this-in-production", env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # Rate Limiting
    rate_limit_requests_per_minute: int = Field(default=100, env="RATE_LIMIT_REQUESTS_PER_MINUTE")
    rate_limit_burst_size: int = Field(default=20, env="RATE_LIMIT_BURST_SIZE")
    
    # Monitoring & Observability
    prometheus_enabled: bool = Field(default=True, env="PROMETHEUS_ENABLED")
    prometheus_port: int = Field(default=9090, env="PROMETHEUS_PORT")
    jaeger_enabled: bool = Field(default=False, env="JAEGER_ENABLED")
    jaeger_endpoint: str = Field(default="http://localhost:14268/api/traces", env="JAEGER_ENDPOINT")
    
    # Multi-Agent Configuration
    max_concurrent_agents: int = Field(default=10, env="MAX_CONCURRENT_AGENTS")
    agent_timeout_seconds: int = Field(default=300, env="AGENT_TIMEOUT_SECONDS")
    reflection_enabled: bool = Field(default=True, env="REFLECTION_ENABLED")
    reflection_threshold: float = Field(default=0.7, env="REFLECTION_THRESHOLD")
    
    # Code Execution Security
    code_execution_enabled: bool = Field(default=True, env="CODE_EXECUTION_ENABLED")
    code_execution_timeout: int = Field(default=60, env="CODE_EXECUTION_TIMEOUT")
    allowed_packages: str = Field(
        default="requests,pandas,numpy,matplotlib,seaborn",
        env="ALLOWED_PACKAGES"
    )
    blocked_imports: str = Field(
        default="os,sys,subprocess,socket",
        env="BLOCKED_IMPORTS"
    )
    
    # Performance Tuning
    worker_processes: int = Field(default=4, env="WORKER_PROCESSES")
    worker_connections: int = Field(default=1000, env="WORKER_CONNECTIONS")
    keepalive_timeout: int = Field(default=5, env="KEEPALIVE_TIMEOUT")
    max_request_size: int = Field(default=10485760, env="MAX_REQUEST_SIZE")  # 10MB
    
    # Feature Flags
    enable_caching: bool = Field(default=True, env="ENABLE_CACHING")
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    enable_tracing: bool = Field(default=False, env="ENABLE_TRACING")
    enable_health_checks: bool = Field(default=True, env="ENABLE_HEALTH_CHECKS")
    enable_cors: bool = Field(default=True, env="ENABLE_CORS")
    
    # CORS Configuration
    cors_origins: str = Field(
        default="http://localhost:3000,http://localhost:8080",
        env="CORS_ORIGINS"
    )
    cors_methods: str = Field(
        default="GET,POST,PUT,DELETE,OPTIONS",
        env="CORS_METHODS"
    )
    cors_headers: str = Field(default="*", env="CORS_HEADERS")
    
    @validator("log_level")
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()
    
    @validator("reflection_threshold")
    def validate_reflection_threshold(cls, v):
        """Validate reflection threshold is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("Reflection threshold must be between 0 and 1")
        return v
    
    @property
    def allowed_packages_list(self) -> List[str]:
        """Get allowed packages as list."""
        return [item.strip() for item in self.allowed_packages.split(",") if item.strip()]

    @property
    def blocked_imports_list(self) -> List[str]:
        """Get blocked imports as list."""
        return [item.strip() for item in self.blocked_imports.split(",") if item.strip()]

    @property
    def cors_origins_list(self) -> List[str]:
        """Get CORS origins as list."""
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]

    @property
    def cors_methods_list(self) -> List[str]:
        """Get CORS methods as list."""
        return [item.strip() for item in self.cors_methods.split(",") if item.strip()]

    @property
    def cors_headers_list(self) -> List[str]:
        """Get CORS headers as list."""
        if self.cors_headers == "*":
            return ["*"]
        return [item.strip() for item in self.cors_headers.split(",") if item.strip()]
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
