"""
Main FastAPI application for Omni-Agent Hub.

This module creates and configures the FastAPI application with all
necessary middleware, routers, and startup/shutdown handlers.
"""

import asyncio
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from .core.config import get_settings
from .core.exceptions import OmniAgentException
from .core.logging import setup_logging, get_logger, RequestLogger
from .api.routes import api_router
from .services.database import DatabaseManager
from .services.redis_manager import RedisManager
from .services.vector_db import VectorDatabaseManager


# Global managers
db_manager: DatabaseManager = None
redis_manager: RedisManager = None
vector_db_manager: VectorDatabaseManager = None

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    settings = get_settings()
    
    # Setup logging
    setup_logging(
        log_level=settings.log_level,
        log_format="json" if not settings.debug else "console",
        enable_colors=settings.debug
    )
    
    logger.info("Starting Omni-Agent Hub", version=settings.app_version)
    
    # Initialize global managers
    global db_manager, redis_manager, vector_db_manager
    
    try:
        # Initialize database
        db_manager = DatabaseManager(settings.database_url)
        await db_manager.initialize()
        logger.info("Database initialized successfully")
        
        # Initialize Redis
        redis_manager = RedisManager(settings.redis_url)
        await redis_manager.initialize()
        logger.info("Redis initialized successfully")
        
        # Initialize Vector Database
        vector_db_manager = VectorDatabaseManager(
            host=settings.milvus_host,
            port=settings.milvus_port,
            user=settings.milvus_user,
            password=settings.milvus_password
        )
        await vector_db_manager.initialize()
        logger.info("Vector database initialized successfully")
        
        logger.info("All services initialized successfully")
        
    except Exception as e:
        logger.error("Failed to initialize services", error=str(e), exc_info=True)
        raise
    
    yield
    
    # Cleanup
    logger.info("Shutting down Omni-Agent Hub")
    
    try:
        if vector_db_manager:
            await vector_db_manager.close()
            logger.info("Vector database closed")
        
        if redis_manager:
            await redis_manager.close()
            logger.info("Redis closed")
        
        if db_manager:
            await db_manager.close()
            logger.info("Database closed")
            
    except Exception as e:
        logger.error("Error during shutdown", error=str(e), exc_info=True)
    
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Advanced Multi-Agent Orchestration System with ReAct, MCP, and Agentic RAG",
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        openapi_url="/openapi.json" if settings.debug else None,
        lifespan=lifespan
    )
    
    # Add middleware
    setup_middleware(app, settings)
    
    # Add routers
    app.include_router(api_router, prefix="/api/v1")
    
    # Add exception handlers
    setup_exception_handlers(app)
    
    # Add request/response middleware
    setup_request_middleware(app)
    
    return app


def setup_middleware(app: FastAPI, settings) -> None:
    """Setup application middleware."""
    
    # CORS middleware
    if settings.enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins_list,
            allow_credentials=True,
            allow_methods=settings.cors_methods_list,
            allow_headers=settings.cors_headers_list,
        )
    
    # Trusted host middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"] if settings.debug else ["localhost", "127.0.0.1"]
    )


def setup_exception_handlers(app: FastAPI) -> None:
    """Setup custom exception handlers."""
    
    @app.exception_handler(OmniAgentException)
    async def omni_agent_exception_handler(request: Request, exc: OmniAgentException):
        """Handle custom Omni-Agent exceptions."""
        correlation_id = getattr(request.state, "correlation_id", "unknown")
        
        logger.error(
            "Application error",
            correlation_id=correlation_id,
            error_code=exc.error_code,
            error_message=exc.message,
            context=exc.context,
            exc_info=True
        )
        
        return JSONResponse(
            status_code=400,
            content=exc.to_dict()
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions."""
        correlation_id = getattr(request.state, "correlation_id", "unknown")
        
        logger.error(
            "Unhandled error",
            correlation_id=correlation_id,
            error_type=type(exc).__name__,
            error_message=str(exc),
            exc_info=True
        )
        
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "An internal server error occurred",
                    "correlation_id": correlation_id
                }
            }
        )


def setup_request_middleware(app: FastAPI) -> None:
    """Setup request/response logging middleware."""
    
    @app.middleware("http")
    async def request_middleware(request: Request, call_next):
        """Log requests and responses with correlation ID."""
        # Generate correlation ID
        correlation_id = str(uuid.uuid4())
        request.state.correlation_id = correlation_id
        
        # Create request logger
        request_logger = RequestLogger(correlation_id)
        
        # Log incoming request
        start_time = asyncio.get_event_loop().time()
        request_logger.log_request(
            method=request.method,
            path=request.url.path,
            query_params=dict(request.query_params),
            headers=dict(request.headers)
        )
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration_ms = (asyncio.get_event_loop().time() - start_time) * 1000
        
        # Log response
        request_logger.log_response(
            status_code=response.status_code,
            duration_ms=duration_ms
        )
        
        # Add correlation ID to response headers
        response.headers["X-Correlation-ID"] = correlation_id
        
        return response


# Create the application instance
app = create_app()


def main():
    """Main entry point for running the application."""
    settings = get_settings()
    
    uvicorn.run(
        "omni_agent_hub.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=1 if settings.debug else settings.worker_processes,
        log_level=settings.log_level.lower(),
        access_log=True,
    )


if __name__ == "__main__":
    main()
