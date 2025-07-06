"""
Custom exceptions for Omni-Agent Hub.

This module defines all custom exceptions used throughout the application
with proper error codes and context information.
"""

from typing import Any, Dict, Optional


class OmniAgentException(Exception):
    """Base exception for all Omni-Agent Hub errors."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "UNKNOWN_ERROR",
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        self.cause = cause
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error": {
                "code": self.error_code,
                "message": self.message,
                "context": self.context,
                "type": self.__class__.__name__
            }
        }


class ConfigurationError(OmniAgentException):
    """Raised when there's a configuration error."""
    
    def __init__(self, message: str, config_key: Optional[str] = None):
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            context={"config_key": config_key} if config_key else {}
        )


class DatabaseError(OmniAgentException):
    """Raised when there's a database operation error."""
    
    def __init__(self, message: str, operation: Optional[str] = None):
        super().__init__(
            message=message,
            error_code="DATABASE_ERROR",
            context={"operation": operation} if operation else {}
        )


class VectorDatabaseError(OmniAgentException):
    """Raised when there's a vector database operation error."""
    
    def __init__(self, message: str, collection: Optional[str] = None):
        super().__init__(
            message=message,
            error_code="VECTOR_DB_ERROR",
            context={"collection": collection} if collection else {}
        )


class AgentError(OmniAgentException):
    """Raised when there's an agent execution error."""
    
    def __init__(
        self,
        message: str,
        agent_name: Optional[str] = None,
        task_id: Optional[str] = None
    ):
        super().__init__(
            message=message,
            error_code="AGENT_ERROR",
            context={
                "agent_name": agent_name,
                "task_id": task_id
            }
        )


class AgentTimeoutError(AgentError):
    """Raised when an agent operation times out."""
    
    def __init__(
        self,
        message: str,
        agent_name: Optional[str] = None,
        timeout_seconds: Optional[int] = None
    ):
        super().__init__(
            message=message,
            agent_name=agent_name
        )
        self.error_code = "AGENT_TIMEOUT"
        self.context["timeout_seconds"] = timeout_seconds


class ToolExecutionError(OmniAgentException):
    """Raised when there's a tool execution error."""
    
    def __init__(
        self,
        message: str,
        tool_name: Optional[str] = None,
        tool_input: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code="TOOL_EXECUTION_ERROR",
            context={
                "tool_name": tool_name,
                "tool_input": tool_input
            }
        )


class MCPError(OmniAgentException):
    """Raised when there's an MCP protocol error."""
    
    def __init__(
        self,
        message: str,
        protocol_version: Optional[str] = None,
        server_endpoint: Optional[str] = None
    ):
        super().__init__(
            message=message,
            error_code="MCP_ERROR",
            context={
                "protocol_version": protocol_version,
                "server_endpoint": server_endpoint
            }
        )


class CodeExecutionError(OmniAgentException):
    """Raised when there's a code execution error."""
    
    def __init__(
        self,
        message: str,
        code_snippet: Optional[str] = None,
        execution_environment: Optional[str] = None
    ):
        super().__init__(
            message=message,
            error_code="CODE_EXECUTION_ERROR",
            context={
                "code_snippet": code_snippet,
                "execution_environment": execution_environment
            }
        )


class SecurityError(OmniAgentException):
    """Raised when there's a security violation."""
    
    def __init__(
        self,
        message: str,
        security_rule: Optional[str] = None,
        attempted_action: Optional[str] = None
    ):
        super().__init__(
            message=message,
            error_code="SECURITY_ERROR",
            context={
                "security_rule": security_rule,
                "attempted_action": attempted_action
            }
        )


class RateLimitError(OmniAgentException):
    """Raised when rate limit is exceeded."""
    
    def __init__(
        self,
        message: str,
        limit: Optional[int] = None,
        window_seconds: Optional[int] = None,
        retry_after: Optional[int] = None
    ):
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_EXCEEDED",
            context={
                "limit": limit,
                "window_seconds": window_seconds,
                "retry_after": retry_after
            }
        )


class ValidationError(OmniAgentException):
    """Raised when there's a validation error."""
    
    def __init__(
        self,
        message: str,
        field_name: Optional[str] = None,
        field_value: Optional[Any] = None
    ):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            context={
                "field_name": field_name,
                "field_value": str(field_value) if field_value is not None else None
            }
        )


class WorkflowError(OmniAgentException):
    """Raised when there's a workflow execution error."""
    
    def __init__(
        self,
        message: str,
        workflow_name: Optional[str] = None,
        step_name: Optional[str] = None,
        workflow_id: Optional[str] = None
    ):
        super().__init__(
            message=message,
            error_code="WORKFLOW_ERROR",
            context={
                "workflow_name": workflow_name,
                "step_name": step_name,
                "workflow_id": workflow_id
            }
        )


class ReflectionError(OmniAgentException):
    """Raised when there's a self-reflection process error."""
    
    def __init__(
        self,
        message: str,
        reflection_type: Optional[str] = None,
        original_output: Optional[str] = None
    ):
        super().__init__(
            message=message,
            error_code="REFLECTION_ERROR",
            context={
                "reflection_type": reflection_type,
                "original_output": original_output
            }
        )


class ExternalServiceError(OmniAgentException):
    """Raised when there's an external service error."""
    
    def __init__(
        self,
        message: str,
        service_name: Optional[str] = None,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None
    ):
        super().__init__(
            message=message,
            error_code="EXTERNAL_SERVICE_ERROR",
            context={
                "service_name": service_name,
                "status_code": status_code,
                "response_body": response_body
            }
        )
