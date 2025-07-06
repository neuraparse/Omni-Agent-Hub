"""
Dependency Injection Container for Omni-Agent Hub.

This module provides a centralized dependency injection system
for managing all services and their dependencies.
"""

from typing import Dict, Any, Optional, TypeVar, Type
from dataclasses import dataclass
import asyncio

from ..core.logging import LoggerMixin
from ..core.config import get_settings


T = TypeVar('T')


@dataclass
class ServiceConfig:
    """Configuration for a service."""
    name: str
    singleton: bool = True
    lazy: bool = True
    dependencies: list = None


class ServiceContainer(LoggerMixin):
    """Centralized dependency injection container."""
    
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, callable] = {}
        self._configs: Dict[str, ServiceConfig] = {}
        self._initialized = False
        
        self.logger.info("Service container initialized")
    
    def register(
        self, 
        name: str, 
        factory: callable, 
        singleton: bool = True,
        lazy: bool = True,
        dependencies: list = None
    ):
        """Register a service factory."""
        self._factories[name] = factory
        self._configs[name] = ServiceConfig(
            name=name,
            singleton=singleton,
            lazy=lazy,
            dependencies=dependencies or []
        )
        
        self.logger.debug(f"Registered service: {name}")
    
    async def get(self, name: str) -> Any:
        """Get a service instance."""
        if name in self._services:
            return self._services[name]
        
        if name not in self._factories:
            raise ValueError(f"Service '{name}' not registered")
        
        # Create instance
        factory = self._factories[name]
        config = self._configs[name]
        
        # Resolve dependencies
        dependencies = {}
        for dep_name in config.dependencies:
            dependencies[dep_name] = await self.get(dep_name)
        
        # Create service instance
        if asyncio.iscoroutinefunction(factory):
            instance = await factory(**dependencies)
        else:
            instance = factory(**dependencies)
        
        # Store if singleton
        if config.singleton:
            self._services[name] = instance
        
        self.logger.debug(f"Created service instance: {name}")
        return instance
    
    async def initialize_all(self):
        """Initialize all registered services."""
        if self._initialized:
            return
        
        self.logger.info("Initializing all services...")
        
        # Initialize in dependency order
        for name in self._factories.keys():
            try:
                await self.get(name)
                self.logger.info(f"✅ {name} initialized successfully")
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize {name}: {str(e)}")
                raise
        
        self._initialized = True
        self.logger.info("🎉 All services initialized successfully!")
    
    async def shutdown_all(self):
        """Shutdown all services."""
        self.logger.info("Shutting down all services...")
        
        for name, service in self._services.items():
            try:
                if hasattr(service, 'close'):
                    if asyncio.iscoroutinefunction(service.close):
                        await service.close()
                    else:
                        service.close()
                self.logger.info(f"✅ {name} shut down successfully")
            except Exception as e:
                self.logger.error(f"❌ Failed to shutdown {name}: {str(e)}")
        
        self._services.clear()
        self._initialized = False
        self.logger.info("🔒 All services shut down")


# Global container instance
container = ServiceContainer()


def register_service(
    name: str,
    singleton: bool = True,
    lazy: bool = True,
    dependencies: list = None
):
    """Decorator for registering services."""
    def decorator(factory):
        container.register(name, factory, singleton, lazy, dependencies)
        return factory
    return decorator


async def get_service(name: str) -> Any:
    """Get a service from the global container."""
    return await container.get(name)


# Service factory functions
@register_service("settings")
async def create_settings():
    """Create settings instance."""
    return get_settings()


@register_service("database", dependencies=["settings"])
async def create_database(settings):
    """Create database manager."""
    from ..services.database import DatabaseManager
    db = DatabaseManager(settings.database_url)
    await db.initialize()
    return db


@register_service("redis", dependencies=["settings"])
async def create_redis(settings):
    """Create Redis manager."""
    from ..services.redis_manager import RedisManager
    redis = RedisManager(settings.redis_url)
    await redis.initialize()
    return redis


@register_service("vector_db", dependencies=["settings"])
async def create_vector_db(settings):
    """Create vector database manager."""
    from ..services.vector_db import VectorDatabaseManager
    vector_db = VectorDatabaseManager(
        host=settings.milvus_host,
        port=settings.milvus_port,
        user=settings.milvus_user,
        password=settings.milvus_password
    )
    await vector_db.initialize()
    return vector_db


@register_service("llm_service")
async def create_llm_service():
    """Create LLM service."""
    from ..services.llm_service import LLMService
    return LLMService()


@register_service("prompt_manager")
async def create_prompt_manager():
    """Create prompt template manager."""
    from ..prompts.prompt_templates import PromptTemplateManager
    return PromptTemplateManager()


@register_service("learning_engine", dependencies=["redis", "database"])
async def create_learning_engine(redis, database):
    """Create adaptive learning engine."""
    from ..learning.adaptive_learning import AdaptiveLearningEngine
    return AdaptiveLearningEngine(redis, database)


@register_service("performance_monitor", dependencies=["redis", "database"])
async def create_performance_monitor(redis, database):
    """Create performance monitor."""
    from ..analytics.performance_monitor import PerformanceMonitor
    return PerformanceMonitor(redis, database)


@register_service("memory_system", dependencies=["redis", "vector_db"])
async def create_memory_system(redis, vector_db):
    """Create context-aware memory system."""
    from ..memory.context_memory import ContextAwareMemorySystem
    return ContextAwareMemorySystem(redis, vector_db)


@register_service("orchestrator", dependencies=[
    "llm_service", "prompt_manager", "learning_engine",
    "database", "redis", "vector_db", "performance_monitor", "memory_system", "kafka", "minio"
])
async def create_orchestrator(
    llm_service, prompt_manager, learning_engine,
    database, redis, vector_db, performance_monitor, memory_system, kafka, minio
):
    """Create ReAct orchestrator with all dependencies."""
    from ..agents.react_orchestrator import ReActOrchestrator

    orchestrator = ReActOrchestrator(
        llm_service=llm_service,
        prompt_manager=prompt_manager,
        learning_engine=learning_engine
    )

    # Inject all dependencies
    orchestrator.db_manager = database
    orchestrator.redis_manager = redis
    orchestrator.vector_db_manager = vector_db
    orchestrator.performance_monitor = performance_monitor
    orchestrator.memory_system = memory_system
    orchestrator.kafka_manager = kafka
    orchestrator.minio_manager = minio

    return orchestrator


# Kafka service for event streaming
@register_service("kafka", dependencies=["settings"])
async def create_kafka(settings):
    """Create Kafka manager for event streaming."""
    from ..services.kafka_manager import KafkaManager
    kafka = KafkaManager(
        bootstrap_servers=settings.kafka_bootstrap_servers,
        client_id="omni-agent-hub",
        group_id="omni-agents"
    )
    await kafka.initialize()

    # Subscribe to default topics
    await kafka.subscribe_to_topic("agent-events", kafka.handle_agent_event)
    await kafka.subscribe_to_topic("system-events", kafka.handle_system_event)

    return kafka


# MinIO service for object storage
@register_service("minio", dependencies=["settings"])
async def create_minio(settings):
    """Create MinIO manager for object storage."""
    from ..services.minio_manager import MinIOManager
    minio = MinIOManager(
        endpoint=settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=settings.minio_secure
    )
    await minio.initialize()
    return minio
