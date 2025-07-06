"""
Omni-Agent Hub: Advanced Multi-Agent Orchestration System

A sophisticated AI agent orchestration platform that combines:
- ReAct (Reasoning and Acting) framework for intelligent decision making
- MCP (Model Context Protocol) for standardized tool integration
- Agentic RAG for context-aware information retrieval
- Multi-agent workflows for complex task decomposition
- Self-reflection mechanisms for quality assurance

Architecture Components:
1. API Gateway (Spring Cloud Gateway equivalent in Python)
2. Orchestrator Service (ReAct Core Engine)
3. CodeAct Runner (Secure code execution environment)
4. ToolHub (MCP-compliant tool integration layer)
5. Self-Reflection Unit (Quality assurance and improvement)
6. Multi-Agent Workflow (Specialized agent coordination)
7. Agentic RAG (Vector-based knowledge retrieval)
"""

__version__ = "1.0.0"
__author__ = "Omni-Agent Team"
__email__ = "team@omni-agent.com"
__description__ = "Advanced Multi-Agent Orchestration System"

# Core imports for easy access
from .core.config import Settings, get_settings
from .core.logging import setup_logging, get_logger
from .core.exceptions import OmniAgentException, ConfigurationError

# Main application factory
from .main import create_app

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__description__",
    "Settings",
    "get_settings",
    "setup_logging",
    "get_logger",
    "OmniAgentException",
    "ConfigurationError",
    "create_app",
]
