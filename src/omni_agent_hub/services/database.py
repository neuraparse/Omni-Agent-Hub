"""
Database manager for PostgreSQL operations.

This module provides async database operations using SQLAlchemy
with connection pooling and transaction management.
"""

import asyncio
from typing import Any, Dict, List, Optional, Union
from contextlib import asynccontextmanager

import asyncpg
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import text

from ..core.logging import LoggerMixin
from ..core.exceptions import DatabaseError


Base = declarative_base()


class DatabaseManager(LoggerMixin):
    """Async database manager for PostgreSQL operations."""
    
    def __init__(self, database_url: str, pool_size: int = 20, max_overflow: int = 30):
        self.database_url = database_url
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.engine = None
        self.session_factory = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize database connection and session factory."""
        try:
            self.logger.info("Initializing database connection")
            
            # Create async engine
            self.engine = create_async_engine(
                self.database_url,
                pool_size=self.pool_size,
                max_overflow=self.max_overflow,
                pool_pre_ping=True,
                echo=False  # Set to True for SQL debugging
            )
            
            # Create session factory
            self.session_factory = async_sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            self._initialized = True

            # Test connection
            await self._test_connection()

            self.logger.info("Database initialized successfully")
            
        except Exception as e:
            self.log_error(e, {"operation": "database_initialization"})
            raise DatabaseError(f"Failed to initialize database: {str(e)}")
    
    async def close(self) -> None:
        """Close database connections."""
        if self.engine:
            await self.engine.dispose()
            self.logger.info("Database connections closed")
    
    @asynccontextmanager
    async def get_db_session(self):
        """Get async database session with automatic cleanup."""
        if not self._initialized:
            raise DatabaseError("Database not initialized")

        async with self.session_factory() as session:
            try:
                yield session
            except Exception as e:
                await session.rollback()
                self.log_error(e, {"operation": "database_session"})
                raise
            finally:
                await session.close()
    
    async def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute a raw SQL query and return results."""
        try:
            async with self.get_db_session() as session:
                result = await session.execute(text(query), parameters or {})
                rows = result.fetchall()

                # Convert rows to dictionaries
                if rows:
                    columns = result.keys()
                    return [dict(zip(columns, row)) for row in rows]
                return []
                
        except Exception as e:
            self.log_error(e, {"operation": "execute_query", "query": query})
            raise DatabaseError(f"Query execution failed: {str(e)}")
    
    async def execute_command(
        self,
        command: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> int:
        """Execute a SQL command (INSERT, UPDATE, DELETE) and return affected rows."""
        try:
            async with self.get_db_session() as session:
                result = await session.execute(text(command), parameters or {})
                await session.commit()
                return result.rowcount
                
        except Exception as e:
            self.log_error(e, {"operation": "execute_command", "command": command})
            raise DatabaseError(f"Command execution failed: {str(e)}")
    
    async def _test_connection(self) -> None:
        """Test database connection."""
        try:
            async with self.get_db_session() as session:
                await session.execute(text("SELECT 1"))
                self.logger.info("Database connection test successful")
        except Exception as e:
            raise DatabaseError(f"Database connection test failed: {str(e)}")
    
    # Agent Sessions Operations
    async def create_session(
        self,
        session_id: str,
        user_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a new agent session."""
        query = """
        INSERT INTO agent_sessions (session_id, user_id, context, status)
        VALUES (:session_id, :user_id, :context, 'active')
        RETURNING id, session_id, user_id, status, created_at, updated_at
        """
        
        try:
            async with self.get_db_session() as db_session:
                result = await db_session.execute(
                    text(query),
                    {
                        "session_id": session_id,
                        "user_id": user_id,
                        "context": context
                    }
                )
                await db_session.commit()
                row = result.fetchone()

                if row:
                    columns = result.keys()
                    return dict(zip(columns, row))

                raise DatabaseError("Failed to create session")
                
        except Exception as e:
            self.log_error(e, {"operation": "create_session", "session_id": session_id})
            raise DatabaseError(f"Session creation failed: {str(e)}")
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session by ID."""
        query = """
        SELECT id, session_id, user_id, status, context, created_at, updated_at
        FROM agent_sessions
        WHERE session_id = :session_id
        """
        
        try:
            results = await self.execute_query(query, {"session_id": session_id})
            return results[0] if results else None
            
        except Exception as e:
            self.log_error(e, {"operation": "get_session", "session_id": session_id})
            raise DatabaseError(f"Session retrieval failed: {str(e)}")
    
    async def update_session_status(
        self,
        session_id: str,
        status: str
    ) -> bool:
        """Update session status."""
        command = """
        UPDATE agent_sessions
        SET status = :status, updated_at = NOW()
        WHERE session_id = :session_id
        """
        
        try:
            affected_rows = await self.execute_command(
                command,
                {"session_id": session_id, "status": status}
            )
            return affected_rows > 0
            
        except Exception as e:
            self.log_error(e, {"operation": "update_session_status", "session_id": session_id})
            raise DatabaseError(f"Session status update failed: {str(e)}")
    
    # Agent Tasks Operations
    async def create_task(
        self,
        session_id: str,
        task_type: str,
        task_data: Dict[str, Any],
        priority: int = 5,
        assigned_agent: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a new agent task."""
        query = """
        INSERT INTO agent_tasks (session_id, task_type, task_data, priority, assigned_agent, status)
        VALUES (
            (SELECT id FROM agent_sessions WHERE session_id = :session_id),
            :task_type, :task_data, :priority, :assigned_agent, 'pending'
        )
        RETURNING id, task_type, task_data, status, priority, created_at
        """
        
        try:
            async with self.get_db_session() as db_session:
                result = await db_session.execute(
                    text(query),
                    {
                        "session_id": session_id,
                        "task_type": task_type,
                        "task_data": task_data,
                        "priority": priority,
                        "assigned_agent": assigned_agent
                    }
                )
                await db_session.commit()
                row = result.fetchone()

                if row:
                    columns = result.keys()
                    return dict(zip(columns, row))

                raise DatabaseError("Failed to create task")
                
        except Exception as e:
            self.log_error(e, {"operation": "create_task", "session_id": session_id})
            raise DatabaseError(f"Task creation failed: {str(e)}")
    
    async def update_task_status(
        self,
        task_id: str,
        status: str,
        result: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None
    ) -> bool:
        """Update task status and result."""
        command = """
        UPDATE agent_tasks
        SET status = :status, result = :result, error_message = :error_message,
            completed_at = CASE WHEN :status IN ('completed', 'failed') THEN NOW() ELSE completed_at END
        WHERE id = :task_id
        """
        
        try:
            affected_rows = await self.execute_command(
                command,
                {
                    "task_id": task_id,
                    "status": status,
                    "result": result,
                    "error_message": error_message
                }
            )
            return affected_rows > 0
            
        except Exception as e:
            self.log_error(e, {"operation": "update_task_status", "task_id": task_id})
            raise DatabaseError(f"Task status update failed: {str(e)}")
    
    async def get_pending_tasks(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get pending tasks ordered by priority."""
        query = """
        SELECT t.id, t.task_type, t.task_data, t.priority, t.assigned_agent,
               s.session_id, s.user_id, t.created_at
        FROM agent_tasks t
        JOIN agent_sessions s ON t.session_id = s.id
        WHERE t.status = 'pending'
        ORDER BY t.priority ASC, t.created_at ASC
        LIMIT :limit
        """
        
        try:
            return await self.execute_query(query, {"limit": limit})
            
        except Exception as e:
            self.log_error(e, {"operation": "get_pending_tasks"})
            raise DatabaseError(f"Pending tasks retrieval failed: {str(e)}")
