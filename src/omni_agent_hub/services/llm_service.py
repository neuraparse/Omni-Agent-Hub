"""
LLM service for AI model integration.

This module provides unified interface for different LLM providers
including OpenAI, Anthropic, and local models.
"""

import asyncio
from typing import Any, Dict, List, Optional, Union
from enum import Enum
import json

from ..core.logging import LoggerMixin
from ..core.exceptions import ExternalServiceError
from ..core.config import get_settings


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


class LLMMessage:
    """Represents a message in LLM conversation."""
    
    def __init__(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        self.role = role  # system, user, assistant
        self.content = content
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            **self.metadata
        }


class LLMResponse:
    """Response from LLM service."""
    
    def __init__(
        self,
        content: str,
        provider: LLMProvider,
        model: str,
        usage: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.content = content
        self.provider = provider
        self.model = model
        self.usage = usage or {}
        self.metadata = metadata or {}


class LLMService(LoggerMixin):
    """Unified LLM service supporting multiple providers."""
    
    def __init__(self):
        self.settings = get_settings()
        self._openai_client = None
        self._anthropic_client = None
        
        # Initialize clients based on available API keys
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize LLM clients based on configuration."""
        try:
            if self.settings.openai_api_key and self.settings.openai_api_key != "your_openai_api_key_here":
                import openai
                self._openai_client = openai.AsyncOpenAI(
                    api_key=self.settings.openai_api_key
                )
                self.logger.info("OpenAI client initialized")
        except ImportError:
            self.logger.warning("OpenAI library not available")
        except Exception as e:
            self.logger.error("Failed to initialize OpenAI client", error=str(e))
        
        try:
            if self.settings.anthropic_api_key and self.settings.anthropic_api_key != "your_anthropic_api_key_here":
                import anthropic
                self._anthropic_client = anthropic.AsyncAnthropic(
                    api_key=self.settings.anthropic_api_key
                )
                self.logger.info("Anthropic client initialized")
        except ImportError:
            self.logger.warning("Anthropic library not available")
        except Exception as e:
            self.logger.error("Failed to initialize Anthropic client", error=str(e))
    
    async def generate_response(
        self,
        messages: List[LLMMessage],
        provider: Optional[LLMProvider] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response using specified LLM provider."""
        
        # Auto-select provider if not specified
        if provider is None:
            provider = self._get_default_provider()
        
        # Auto-select model if not specified
        if model is None:
            model = self._get_default_model(provider)
        
        self.logger.info(
            "Generating LLM response",
            provider=provider,
            model=model,
            message_count=len(messages)
        )
        
        try:
            if provider == LLMProvider.OPENAI:
                return await self._generate_openai_response(
                    messages, model, temperature, max_tokens, **kwargs
                )
            elif provider == LLMProvider.ANTHROPIC:
                return await self._generate_anthropic_response(
                    messages, model, temperature, max_tokens, **kwargs
                )
            elif provider == LLMProvider.LOCAL:
                return await self._generate_local_response(
                    messages, model, temperature, max_tokens, **kwargs
                )
            else:
                raise ExternalServiceError(f"Unsupported LLM provider: {provider}")
                
        except Exception as e:
            self.log_error(e, {"provider": provider, "model": model})
            raise ExternalServiceError(f"LLM generation failed: {str(e)}")
    
    async def _generate_openai_response(
        self,
        messages: List[LLMMessage],
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        **kwargs
    ) -> LLMResponse:
        """Generate response using OpenAI API."""
        if not self._openai_client:
            raise ExternalServiceError("OpenAI client not initialized")
        
        # Convert messages to OpenAI format
        openai_messages = [msg.to_dict() for msg in messages]
        
        response = await self._openai_client.chat.completions.create(
            model=model,
            messages=openai_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        content = response.choices[0].message.content
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
        
        return LLMResponse(
            content=content,
            provider=LLMProvider.OPENAI,
            model=model,
            usage=usage,
            metadata={"response_id": response.id}
        )
    
    async def _generate_anthropic_response(
        self,
        messages: List[LLMMessage],
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        **kwargs
    ) -> LLMResponse:
        """Generate response using Anthropic API."""
        if not self._anthropic_client:
            raise ExternalServiceError("Anthropic client not initialized")
        
        # Convert messages to Anthropic format
        system_message = None
        anthropic_messages = []
        
        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                anthropic_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        response = await self._anthropic_client.messages.create(
            model=model,
            messages=anthropic_messages,
            system=system_message,
            temperature=temperature,
            max_tokens=max_tokens or 1000,
            **kwargs
        )
        
        content = response.content[0].text
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens
        }
        
        return LLMResponse(
            content=content,
            provider=LLMProvider.ANTHROPIC,
            model=model,
            usage=usage,
            metadata={"response_id": response.id}
        )
    
    async def _generate_local_response(
        self,
        messages: List[LLMMessage],
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        **kwargs
    ) -> LLMResponse:
        """Generate response using local model (fallback only)."""
        # This should only be used as a last resort fallback
        raise ExternalServiceError("No LLM providers available. Please configure OpenAI or Anthropic API keys.")
    
    def _get_default_provider(self) -> LLMProvider:
        """Get default LLM provider based on available clients."""
        if self._openai_client:
            return LLMProvider.OPENAI
        elif self._anthropic_client:
            return LLMProvider.ANTHROPIC
        else:
            raise ExternalServiceError("No LLM providers configured. Please add OpenAI or Anthropic API keys.")
    
    def _get_default_model(self, provider: LLMProvider) -> str:
        """Get default model for provider."""
        if provider == LLMProvider.OPENAI:
            return self.settings.openai_model
        elif provider == LLMProvider.ANTHROPIC:
            return self.settings.anthropic_model
        else:
            return "local-model"
    
    async def generate_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None
    ) -> List[List[float]]:
        """Generate embeddings for texts."""
        if not self._openai_client:
            raise ExternalServiceError("OpenAI client required for embeddings")
        
        model = model or self.settings.openai_embedding_model
        
        response = await self._openai_client.embeddings.create(
            model=model,
            input=texts
        )
        
        embeddings = [data.embedding for data in response.data]
        
        self.logger.info(
            "Generated embeddings",
            text_count=len(texts),
            model=model,
            embedding_dim=len(embeddings[0]) if embeddings else 0
        )
        
        return embeddings
    
    def get_available_providers(self) -> List[LLMProvider]:
        """Get list of available LLM providers."""
        providers = []
        
        if self._openai_client:
            providers.append(LLMProvider.OPENAI)
        if self._anthropic_client:
            providers.append(LLMProvider.ANTHROPIC)
        
        # Local is always available as fallback
        providers.append(LLMProvider.LOCAL)
        
        return providers
    
    def get_provider_status(self) -> Dict[str, Any]:
        """Get status of all LLM providers."""
        return {
            "openai": {
                "available": self._openai_client is not None,
                "model": self.settings.openai_model if self._openai_client else None
            },
            "anthropic": {
                "available": self._anthropic_client is not None,
                "model": self.settings.anthropic_model if self._anthropic_client else None
            },
            "local": {
                "available": True,
                "model": "placeholder"
            }
        }
