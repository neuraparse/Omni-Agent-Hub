"""
Structured logging configuration for Omni-Agent Hub.

This module provides centralized logging configuration with structured
logging support using structlog for better observability.
"""

import logging
import sys
from typing import Any, Dict, Optional

import structlog
from structlog.stdlib import LoggerFactory


def setup_logging(
    log_level: str = "INFO",
    log_format: str = "json",
    enable_colors: bool = True,
) -> None:
    """
    Setup structured logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Log format ('json' or 'console')
        enable_colors: Enable colored output for console format
    """
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )
    
    # Configure structlog processors
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    
    if log_format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        if enable_colors and sys.stdout.isatty():
            processors.append(structlog.dev.ConsoleRenderer(colors=True))
        else:
            processors.append(structlog.dev.ConsoleRenderer(colors=False))
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""
    
    @property
    def logger(self) -> structlog.stdlib.BoundLogger:
        """Get logger for this class."""
        if not hasattr(self, "_logger"):
            self._logger = get_logger(self.__class__.__module__)
        return self._logger
    
    def log_method_call(
        self,
        method_name: str,
        **kwargs: Any
    ) -> None:
        """Log method call with parameters."""
        self.logger.debug(
            "Method called",
            method=method_name,
            class_name=self.__class__.__name__,
            **kwargs
        )
    
    def log_performance(
        self,
        operation: str,
        duration_ms: float,
        **kwargs: Any
    ) -> None:
        """Log performance metrics."""
        self.logger.info(
            "Performance metric",
            operation=operation,
            duration_ms=duration_ms,
            **kwargs
        )
    
    def log_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        """Log error with context."""
        self.logger.error(
            "Error occurred",
            error_type=type(error).__name__,
            error_message=str(error),
            context=context or {},
            **kwargs,
            exc_info=True
        )


class RequestLogger:
    """Logger for HTTP requests with correlation IDs."""
    
    def __init__(self, correlation_id: str):
        self.correlation_id = correlation_id
        self.logger = get_logger("request")
    
    def log_request(
        self,
        method: str,
        path: str,
        user_id: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """Log incoming request."""
        self.logger.info(
            "Request received",
            correlation_id=self.correlation_id,
            method=method,
            path=path,
            user_id=user_id,
            **kwargs
        )
    
    def log_response(
        self,
        status_code: int,
        duration_ms: float,
        **kwargs: Any
    ) -> None:
        """Log outgoing response."""
        self.logger.info(
            "Response sent",
            correlation_id=self.correlation_id,
            status_code=status_code,
            duration_ms=duration_ms,
            **kwargs
        )
    
    def log_agent_action(
        self,
        agent_name: str,
        action: str,
        result: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """Log agent action."""
        self.logger.info(
            "Agent action",
            correlation_id=self.correlation_id,
            agent_name=agent_name,
            action=action,
            result=result,
            **kwargs
        )


class MetricsLogger:
    """Logger for system metrics and monitoring."""
    
    def __init__(self):
        self.logger = get_logger("metrics")
    
    def log_agent_performance(
        self,
        agent_type: str,
        task_type: str,
        success: bool,
        duration_ms: float,
        **kwargs: Any
    ) -> None:
        """Log agent performance metrics."""
        self.logger.info(
            "Agent performance",
            metric_type="agent_performance",
            agent_type=agent_type,
            task_type=task_type,
            success=success,
            duration_ms=duration_ms,
            **kwargs
        )
    
    def log_system_health(
        self,
        component: str,
        status: str,
        details: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        """Log system health metrics."""
        self.logger.info(
            "System health",
            metric_type="system_health",
            component=component,
            status=status,
            details=details or {},
            **kwargs
        )
    
    def log_resource_usage(
        self,
        resource_type: str,
        usage_value: float,
        unit: str,
        **kwargs: Any
    ) -> None:
        """Log resource usage metrics."""
        self.logger.info(
            "Resource usage",
            metric_type="resource_usage",
            resource_type=resource_type,
            usage_value=usage_value,
            unit=unit,
            **kwargs
        )
