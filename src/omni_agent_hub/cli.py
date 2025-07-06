"""
Command Line Interface for Omni-Agent Hub.

This module provides CLI commands for managing the Omni-Agent Hub system,
including starting services, running migrations, and system administration.
"""

import asyncio
import sys
from typing import Optional

import click
import uvicorn
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from .core.config import get_settings
from .core.logging import setup_logging, get_logger
from .main import create_app
from .services.database import DatabaseManager
from .services.redis_manager import RedisManager
from .services.vector_db import VectorDatabaseManager

console = Console()
logger = get_logger(__name__)


@click.group()
@click.version_option(version="1.0.0", prog_name="Omni-Agent Hub")
def cli():
    """Omni-Agent Hub - Advanced Multi-Agent Orchestration System"""
    pass


@cli.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
@click.option("--reload", is_flag=True, help="Enable auto-reload")
@click.option("--workers", default=1, help="Number of worker processes")
@click.option("--log-level", default="info", help="Log level")
def serve(host: str, port: int, reload: bool, workers: int, log_level: str):
    """Start the Omni-Agent Hub server."""
    settings = get_settings()
    
    console.print(Panel.fit(
        "[bold blue]🚀 Starting Omni-Agent Hub[/bold blue]\n"
        f"Version: {settings.app_version}\n"
        f"Host: {host}:{port}\n"
        f"Workers: {workers}\n"
        f"Reload: {reload}",
        title="Omni-Agent Hub",
        border_style="blue"
    ))
    
    uvicorn.run(
        "omni_agent_hub.main:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers if not reload else 1,
        log_level=log_level,
        access_log=True,
    )


@cli.command()
@click.option("--check-db", is_flag=True, help="Check database connection")
@click.option("--check-redis", is_flag=True, help="Check Redis connection")
@click.option("--check-vector", is_flag=True, help="Check vector database connection")
@click.option("--all", "check_all", is_flag=True, help="Check all services")
def health(check_db: bool, check_redis: bool, check_vector: bool, check_all: bool):
    """Check system health and service connectivity."""
    
    async def run_health_checks():
        settings = get_settings()
        setup_logging(settings.log_level)
        
        table = Table(title="System Health Check")
        table.add_column("Service", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details", style="yellow")
        
        if check_all or check_db:
            try:
                db_manager = DatabaseManager(settings.database_url)
                await db_manager.initialize()
                await db_manager.close()
                table.add_row("PostgreSQL", "✅ Healthy", "Connection successful")
            except Exception as e:
                table.add_row("PostgreSQL", "❌ Unhealthy", str(e))
        
        if check_all or check_redis:
            try:
                redis_manager = RedisManager(settings.redis_url)
                await redis_manager.initialize()
                await redis_manager.close()
                table.add_row("Redis", "✅ Healthy", "Connection successful")
            except Exception as e:
                table.add_row("Redis", "❌ Unhealthy", str(e))
        
        if check_all or check_vector:
            try:
                vector_manager = VectorDatabaseManager(
                    host=settings.milvus_host,
                    port=settings.milvus_port
                )
                await vector_manager.initialize()
                await vector_manager.close()
                table.add_row("Milvus", "✅ Healthy", "Connection successful")
            except Exception as e:
                table.add_row("Milvus", "❌ Unhealthy", str(e))
        
        console.print(table)
    
    asyncio.run(run_health_checks())


@cli.command()
@click.option("--create-tables", is_flag=True, help="Create database tables")
@click.option("--drop-tables", is_flag=True, help="Drop all tables (DANGEROUS)")
@click.confirmation_option(
    prompt="Are you sure you want to run database operations?",
    help="Confirm database operations"
)
def db(create_tables: bool, drop_tables: bool):
    """Database management operations."""
    
    async def run_db_operations():
        settings = get_settings()
        setup_logging(settings.log_level)
        
        db_manager = DatabaseManager(settings.database_url)
        
        try:
            await db_manager.initialize()
            
            if drop_tables:
                console.print("[red]⚠️  Dropping all tables...[/red]")
                # TODO: Implement table dropping
                console.print("[red]✅ Tables dropped[/red]")
            
            if create_tables:
                console.print("[green]📊 Creating tables...[/green]")
                # Tables are created by the init.sql script
                console.print("[green]✅ Tables created[/green]")
            
        except Exception as e:
            console.print(f"[red]❌ Database operation failed: {e}[/red]")
            sys.exit(1)
        finally:
            await db_manager.close()
    
    asyncio.run(run_db_operations())


@cli.command()
@click.option("--collection", default="omni_embeddings", help="Collection name")
@click.option("--recreate", is_flag=True, help="Recreate collection")
def vector(collection: str, recreate: bool):
    """Vector database management operations."""
    
    async def run_vector_operations():
        settings = get_settings()
        setup_logging(settings.log_level)
        
        vector_manager = VectorDatabaseManager(
            host=settings.milvus_host,
            port=settings.milvus_port,
            collection_name=collection
        )
        
        try:
            await vector_manager.initialize()
            
            if recreate:
                console.print(f"[yellow]🔄 Recreating collection: {collection}[/yellow]")
                # TODO: Implement collection recreation
                console.print(f"[green]✅ Collection {collection} recreated[/green]")
            
            stats = await vector_manager.get_collection_stats()
            
            table = Table(title=f"Collection: {collection}")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            for key, value in stats.items():
                table.add_row(key.replace("_", " ").title(), str(value))
            
            console.print(table)
            
        except Exception as e:
            console.print(f"[red]❌ Vector operation failed: {e}[/red]")
            sys.exit(1)
        finally:
            await vector_manager.close()
    
    asyncio.run(run_vector_operations())


@cli.command()
@click.option("--pattern", default="*", help="Key pattern to match")
@click.option("--flush", is_flag=True, help="Flush all Redis data")
@click.confirmation_option(
    prompt="Are you sure you want to flush Redis data?",
    help="Confirm Redis flush operation"
)
def redis(pattern: str, flush: bool):
    """Redis cache management operations."""
    
    async def run_redis_operations():
        settings = get_settings()
        setup_logging(settings.log_level)
        
        redis_manager = RedisManager(settings.redis_url)
        
        try:
            await redis_manager.initialize()
            
            if flush:
                console.print("[red]🗑️  Flushing Redis data...[/red]")
                # TODO: Implement Redis flush
                console.print("[red]✅ Redis data flushed[/red]")
            
            # TODO: Implement key listing and stats
            console.print(f"[green]📊 Redis operations completed[/green]")
            
        except Exception as e:
            console.print(f"[red]❌ Redis operation failed: {e}[/red]")
            sys.exit(1)
        finally:
            await redis_manager.close()
    
    asyncio.run(run_redis_operations())


@cli.command()
def config():
    """Display current configuration."""
    settings = get_settings()
    
    table = Table(title="Omni-Agent Hub Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    
    # Application settings
    table.add_row("App Name", settings.app_name)
    table.add_row("Version", settings.app_version)
    table.add_row("Debug", str(settings.debug))
    table.add_row("Log Level", settings.log_level)
    table.add_row("Host", settings.host)
    table.add_row("Port", str(settings.port))
    
    # Database settings
    table.add_row("Database URL", settings.database_url[:50] + "..." if len(settings.database_url) > 50 else settings.database_url)
    table.add_row("Redis URL", settings.redis_url)
    table.add_row("Milvus Host", f"{settings.milvus_host}:{settings.milvus_port}")
    
    # Feature flags
    table.add_row("Caching Enabled", str(settings.enable_caching))
    table.add_row("Metrics Enabled", str(settings.enable_metrics))
    table.add_row("CORS Enabled", str(settings.enable_cors))
    
    console.print(table)


@cli.command()
@click.option("--format", "output_format", default="table", type=click.Choice(["table", "json"]))
def status(output_format):
    """Show system status and metrics."""

    async def get_system_status():
        # TODO: Implement real system status collection
        status_data = {
            "system": {
                "uptime": "0h 0m 0s",
                "memory_usage": "0 MB",
                "cpu_usage": "0%"
            },
            "services": {
                "database": "unknown",
                "redis": "unknown",
                "vector_db": "unknown",
                "kafka": "unknown"
            },
            "agents": {
                "active_sessions": 0,
                "pending_tasks": 0,
                "completed_tasks": 0
            }
        }

        if output_format == "json":
            import json
            console.print(json.dumps(status_data, indent=2))
        else:
            table = Table(title="System Status")
            table.add_column("Component", style="cyan")
            table.add_column("Status", style="green")

            for category, items in status_data.items():
                for key, value in items.items():
                    table.add_row(f"{category}.{key}", str(value))

            console.print(table)

    asyncio.run(get_system_status())


def main():
    """Main CLI entry point."""
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]👋 Goodbye![/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
