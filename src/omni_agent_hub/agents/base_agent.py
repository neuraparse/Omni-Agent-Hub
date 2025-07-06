"""
Base agent class for all AI agents in Omni-Agent Hub.

This module provides the foundation for all agent implementations
with common functionality like LLM integration, memory management,
and tool usage.
"""

import asyncio
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from ..core.logging import LoggerMixin
from ..core.exceptions import AgentError, AgentTimeoutError
from ..services.redis_manager import RedisManager
from ..services.database import DatabaseManager


class AgentCapability:
    """Represents a capability that an agent can perform."""
    
    def __init__(self, name: str, description: str, parameters: Dict[str, Any]):
        self.name = name
        self.description = description
        self.parameters = parameters


class AgentContext:
    """Context object passed between agents and tools."""
    
    def __init__(
        self,
        session_id: str,
        user_id: str,
        correlation_id: Optional[str] = None,
        memory: Optional[Dict[str, Any]] = None
    ):
        self.session_id = session_id
        self.user_id = user_id
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.memory = memory or {}
        self.created_at = datetime.utcnow()
        
        # Agent execution tracking
        self.agent_chain: List[str] = []
        self.tool_calls: List[Dict[str, Any]] = []
        self.reflections: List[Dict[str, Any]] = []


class AgentResult:
    """Result object returned by agent execution."""
    
    def __init__(
        self,
        success: bool,
        content: str,
        agent_name: str,
        context: AgentContext,
        metadata: Optional[Dict[str, Any]] = None,
        artifacts: Optional[List[Dict[str, Any]]] = None,
        confidence: Optional[float] = None
    ):
        self.success = success
        self.content = content
        self.agent_name = agent_name
        self.context = context
        self.metadata = metadata or {}
        self.artifacts = artifacts or []
        self.confidence = confidence
        self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for API responses."""
        return {
            "success": self.success,
            "content": self.content,
            "agent_name": self.agent_name,
            "session_id": self.context.session_id,
            "correlation_id": self.context.correlation_id,
            "metadata": self.metadata,
            "artifacts": self.artifacts,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat()
        }


class BaseAgent(ABC, LoggerMixin):
    """Base class for all AI agents."""
    
    def __init__(
        self,
        name: str,
        description: str,
        capabilities: List[AgentCapability],
        redis_manager: Optional[RedisManager] = None,
        db_manager: Optional[DatabaseManager] = None,
        timeout_seconds: int = 300
    ):
        self.name = name
        self.description = description
        self.capabilities = {cap.name: cap for cap in capabilities}
        self.redis_manager = redis_manager
        self.db_manager = db_manager
        self.timeout_seconds = timeout_seconds
        
        # Agent state
        self.is_busy = False
        self.current_task_id: Optional[str] = None
        
        self.logger.info(f"Agent {self.name} initialized", capabilities=list(self.capabilities.keys()))
    
    @abstractmethod
    async def execute(self, task: str, context: AgentContext) -> AgentResult:
        """Execute a task and return the result."""
        pass
    
    async def execute_with_timeout(self, task: str, context: AgentContext) -> AgentResult:
        """Execute task with timeout protection."""
        try:
            self.is_busy = True
            self.current_task_id = context.correlation_id
            
            # Add this agent to the execution chain
            context.agent_chain.append(self.name)
            
            self.logger.info(
                f"Agent {self.name} starting task",
                task_preview=task[:100],
                correlation_id=context.correlation_id
            )
            
            # Execute with timeout
            result = await asyncio.wait_for(
                self.execute(task, context),
                timeout=self.timeout_seconds
            )
            
            self.logger.info(
                f"Agent {self.name} completed task",
                success=result.success,
                correlation_id=context.correlation_id
            )
            
            return result
            
        except asyncio.TimeoutError:
            error_msg = f"Agent {self.name} timed out after {self.timeout_seconds} seconds"
            self.log_error(
                AgentTimeoutError(error_msg, self.name, self.timeout_seconds),
                {"correlation_id": context.correlation_id}
            )
            return AgentResult(
                success=False,
                content=error_msg,
                agent_name=self.name,
                context=context,
                metadata={"error_type": "timeout"}
            )
            
        except Exception as e:
            error_msg = f"Agent {self.name} failed: {str(e)}"
            self.log_error(e, {"correlation_id": context.correlation_id})
            return AgentResult(
                success=False,
                content=error_msg,
                agent_name=self.name,
                context=context,
                metadata={"error_type": type(e).__name__, "error_details": str(e)}
            )
            
        finally:
            self.is_busy = False
            self.current_task_id = None
    
    async def store_memory(self, key: str, value: Any, context: AgentContext) -> bool:
        """Store information in agent memory."""
        if not self.redis_manager:
            return False
        
        try:
            memory_key = f"agent_memory:{context.session_id}:{self.name}:{key}"
            await self.redis_manager.set(memory_key, value, expire=3600)  # 1 hour
            
            self.logger.debug(
                f"Agent {self.name} stored memory",
                key=key,
                correlation_id=context.correlation_id
            )
            return True
            
        except Exception as e:
            self.log_error(e, {"operation": "store_memory", "key": key})
            return False
    
    async def retrieve_memory(self, key: str, context: AgentContext) -> Optional[Any]:
        """Retrieve information from agent memory."""
        if not self.redis_manager:
            return None
        
        try:
            memory_key = f"agent_memory:{context.session_id}:{self.name}:{key}"
            value = await self.redis_manager.get(memory_key)
            
            if value is not None:
                self.logger.debug(
                    f"Agent {self.name} retrieved memory",
                    key=key,
                    correlation_id=context.correlation_id
                )
            
            return value
            
        except Exception as e:
            self.log_error(e, {"operation": "retrieve_memory", "key": key})
            return None
    
    async def log_tool_call(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_output: Any,
        context: AgentContext,
        execution_time_ms: Optional[int] = None
    ) -> None:
        """Log a tool call for observability."""
        tool_call = {
            "tool_name": tool_name,
            "input": tool_input,
            "output": str(tool_output)[:1000],  # Truncate long outputs
            "execution_time_ms": execution_time_ms,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        context.tool_calls.append(tool_call)
        
        self.logger.info(
            f"Agent {self.name} used tool",
            tool_name=tool_name,
            execution_time_ms=execution_time_ms,
            correlation_id=context.correlation_id
        )
    
    async def add_reflection(
        self,
        reflection_type: str,
        content: str,
        confidence: float,
        context: AgentContext
    ) -> None:
        """Add a self-reflection entry."""
        reflection = {
            "type": reflection_type,
            "content": content,
            "confidence": confidence,
            "agent": self.name,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        context.reflections.append(reflection)
        
        self.logger.info(
            f"Agent {self.name} added reflection",
            reflection_type=reflection_type,
            confidence=confidence,
            correlation_id=context.correlation_id
        )
    
    def can_handle(self, capability: str) -> bool:
        """Check if agent can handle a specific capability."""
        return capability in self.capabilities
    
    def get_capability_info(self, capability: str) -> Optional[AgentCapability]:
        """Get information about a specific capability."""
        return self.capabilities.get(capability)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        return {
            "name": self.name,
            "description": self.description,
            "is_busy": self.is_busy,
            "current_task_id": self.current_task_id,
            "capabilities": list(self.capabilities.keys()),
            "timeout_seconds": self.timeout_seconds
        }
