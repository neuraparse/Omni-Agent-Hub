"""
Pydantic models for API requests and responses.

This module defines all the data models used in the API endpoints
with proper validation and documentation.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum

from pydantic import BaseModel, Field, validator


class TaskType(str, Enum):
    """Available task types."""
    CHAT = "chat"
    CODE_GENERATION = "code_generation"
    DATA_ANALYSIS = "data_analysis"
    WEB_SEARCH = "web_search"
    DOCUMENT_ANALYSIS = "document_analysis"
    WORKFLOW_EXECUTION = "workflow_execution"


class AgentType(str, Enum):
    """Available agent types."""
    ORCHESTRATOR = "orchestrator"
    CODE_AGENT = "code_agent"
    SEARCH_AGENT = "search_agent"
    ANALYSIS_AGENT = "analysis_agent"
    QA_AGENT = "qa_agent"
    ARCHITECT_AGENT = "architect_agent"


class SessionStatus(str, Enum):
    """Session status values."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    EXPIRED = "expired"
    TERMINATED = "terminated"


class TaskStatus(str, Enum):
    """Task status values."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Request Models
class SessionRequest(BaseModel):
    """Request model for creating a new session."""
    user_id: str = Field(..., description="Unique user identifier")
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Initial context for the session"
    )
    preferences: Optional[Dict[str, Any]] = Field(
        default=None,
        description="User preferences and settings"
    )


class AgentRequest(BaseModel):
    """Request model for agent interactions."""
    message: str = Field(..., description="User message or query")
    session_id: str = Field(..., description="Session identifier")
    agent_type: Optional[AgentType] = Field(
        default=AgentType.ORCHESTRATOR,
        description="Preferred agent type for handling the request"
    )
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional context for the request"
    )
    tools: Optional[List[str]] = Field(
        default=None,
        description="Specific tools to use for this request"
    )
    max_iterations: Optional[int] = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of agent iterations"
    )
    
    @validator("message")
    def validate_message(cls, v):
        """Validate message is not empty."""
        if not v.strip():
            raise ValueError("Message cannot be empty")
        return v.strip()


class TaskRequest(BaseModel):
    """Request model for task execution."""
    task_type: TaskType = Field(..., description="Type of task to execute")
    session_id: str = Field(..., description="Session identifier")
    parameters: Dict[str, Any] = Field(..., description="Task parameters")
    priority: Optional[int] = Field(
        default=5,
        ge=1,
        le=10,
        description="Task priority (1=highest, 10=lowest)"
    )
    timeout_seconds: Optional[int] = Field(
        default=300,
        ge=10,
        le=3600,
        description="Task timeout in seconds"
    )


# Response Models
class HealthResponse(BaseModel):
    """Response model for health checks."""
    status: str = Field(..., description="Service health status")
    version: str = Field(..., description="Application version")
    timestamp: str = Field(..., description="Response timestamp")


class SessionResponse(BaseModel):
    """Response model for session operations."""
    session_id: str = Field(..., description="Unique session identifier")
    user_id: str = Field(..., description="User identifier")
    status: SessionStatus = Field(..., description="Current session status")
    created_at: str = Field(..., description="Session creation timestamp")
    last_activity: Optional[str] = Field(
        default=None,
        description="Last activity timestamp"
    )
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Session context"
    )


class AgentResponse(BaseModel):
    """Response model for agent interactions."""
    response: str = Field(..., description="Agent response message")
    session_id: str = Field(..., description="Session identifier")
    agent_type: AgentType = Field(..., description="Agent that handled the request")
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence score of the response"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional response metadata"
    )
    tools_used: Optional[List[str]] = Field(
        default=None,
        description="Tools used to generate the response"
    )
    reflection_feedback: Optional[str] = Field(
        default=None,
        description="Self-reflection feedback on the response"
    )


class TaskResponse(BaseModel):
    """Response model for task execution."""
    task_id: str = Field(..., description="Unique task identifier")
    status: TaskStatus = Field(..., description="Current task status")
    result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Task execution result"
    )
    session_id: str = Field(..., description="Session identifier")
    created_at: str = Field(..., description="Task creation timestamp")
    started_at: Optional[str] = Field(
        default=None,
        description="Task start timestamp"
    )
    completed_at: Optional[str] = Field(
        default=None,
        description="Task completion timestamp"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if task failed"
    )
    agent_assignments: Optional[List[str]] = Field(
        default=None,
        description="Agents assigned to this task"
    )


# Workflow Models
class WorkflowStep(BaseModel):
    """Model for workflow step."""
    name: str = Field(..., description="Step name")
    status: TaskStatus = Field(..., description="Step status")
    agent_type: Optional[AgentType] = Field(
        default=None,
        description="Agent assigned to this step"
    )
    input_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Input data for the step"
    )
    output_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Output data from the step"
    )
    started_at: Optional[str] = Field(
        default=None,
        description="Step start timestamp"
    )
    completed_at: Optional[str] = Field(
        default=None,
        description="Step completion timestamp"
    )


class WorkflowResponse(BaseModel):
    """Response model for workflow operations."""
    workflow_id: str = Field(..., description="Unique workflow identifier")
    workflow_name: str = Field(..., description="Workflow name")
    status: TaskStatus = Field(..., description="Overall workflow status")
    session_id: str = Field(..., description="Session identifier")
    steps: List[WorkflowStep] = Field(..., description="Workflow steps")
    progress: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Workflow progress (0.0 to 1.0)"
    )
    created_at: str = Field(..., description="Workflow creation timestamp")
    completed_at: Optional[str] = Field(
        default=None,
        description="Workflow completion timestamp"
    )


# Tool Models
class ToolParameter(BaseModel):
    """Model for tool parameter definition."""
    name: str = Field(..., description="Parameter name")
    type: str = Field(..., description="Parameter type")
    description: str = Field(..., description="Parameter description")
    required: bool = Field(default=True, description="Whether parameter is required")
    default_value: Optional[Any] = Field(
        default=None,
        description="Default parameter value"
    )


class ToolDefinition(BaseModel):
    """Model for tool definition."""
    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    category: str = Field(..., description="Tool category")
    parameters: List[ToolParameter] = Field(..., description="Tool parameters")
    mcp_compatible: bool = Field(
        default=True,
        description="Whether tool supports MCP protocol"
    )


class ToolExecutionResult(BaseModel):
    """Model for tool execution result."""
    tool_name: str = Field(..., description="Executed tool name")
    status: str = Field(..., description="Execution status")
    result: Optional[Any] = Field(
        default=None,
        description="Tool execution result"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if execution failed"
    )
    execution_time_ms: Optional[int] = Field(
        default=None,
        description="Execution time in milliseconds"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional execution metadata"
    )


# Knowledge and RAG Models
class KnowledgeSearchResult(BaseModel):
    """Model for knowledge search result."""
    content: str = Field(..., description="Retrieved content")
    similarity: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Similarity score"
    )
    source: str = Field(..., description="Content source")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata"
    )


class KnowledgeSearchResponse(BaseModel):
    """Response model for knowledge search."""
    query: str = Field(..., description="Search query")
    results: List[KnowledgeSearchResult] = Field(..., description="Search results")
    total_results: int = Field(..., description="Total number of results")
    search_time_ms: Optional[int] = Field(
        default=None,
        description="Search time in milliseconds"
    )
