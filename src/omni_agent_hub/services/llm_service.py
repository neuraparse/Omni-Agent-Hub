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
    """Supported LLM providers (2025 Updated)."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
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
    """Unified LLM service supporting multiple providers (2025 Updated)."""

    def __init__(self):
        self.settings = get_settings()
        self._openai_client = None
        self._anthropic_client = None
        self._google_client = None

        # Track model availability and performance
        self._model_health = {}
        self._fallback_attempts = {}

        # Initialize clients based on available API keys
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize LLM clients based on configuration (2025 Updated)."""
        # OpenAI Client
        try:
            if self.settings.openai_api_key and self.settings.openai_api_key != "your_openai_api_key_here":
                import openai
                self._openai_client = openai.AsyncOpenAI(
                    api_key=self.settings.openai_api_key
                )
                self._model_health["openai"] = True
                self.logger.info("OpenAI client initialized", models=[
                    self.settings.openai_model,
                    self.settings.openai_fallback_model
                ])
        except ImportError:
            self.logger.warning("OpenAI library not available")
            self._model_health["openai"] = False
        except Exception as e:
            self.logger.error("Failed to initialize OpenAI client", error=str(e))
            self._model_health["openai"] = False

        # Anthropic Client
        try:
            if self.settings.anthropic_api_key and self.settings.anthropic_api_key != "your_anthropic_api_key_here":
                import anthropic
                self._anthropic_client = anthropic.AsyncAnthropic(
                    api_key=self.settings.anthropic_api_key
                )
                self._model_health["anthropic"] = True
                self.logger.info("Anthropic client initialized", models=[
                    self.settings.anthropic_model,
                    self.settings.anthropic_fallback_model
                ])
        except ImportError:
            self.logger.warning("Anthropic library not available")
            self._model_health["anthropic"] = False
        except Exception as e:
            self.logger.error("Failed to initialize Anthropic client", error=str(e))
            self._model_health["anthropic"] = False

        # Google Client (Optional)
        try:
            if self.settings.google_api_key and self.settings.google_api_key != "your_google_api_key_here":
                # TODO: Initialize Google Gemini client when available
                self._model_health["google"] = False
                self.logger.info("Google Gemini client placeholder")
        except Exception as e:
            self.logger.error("Failed to initialize Google client", error=str(e))
            self._model_health["google"] = False
    
    async def generate_response(
        self,
        messages: List[LLMMessage],
        provider: Optional[LLMProvider] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        task_type: str = "general",
        **kwargs
    ) -> LLMResponse:
        """Generate response using specified LLM provider with smart fallback (2025)."""

        # Auto-select provider if not specified
        if provider is None:
            provider = self._get_optimal_provider(task_type)

        # Auto-select model if not specified
        if model is None:
            model = self._get_optimal_model(provider, task_type)

        self.logger.info(
            "Generating LLM response",
            provider=provider,
            model=model,
            task_type=task_type,
            message_count=len(messages)
        )

        # Try primary model first
        try:
            return await self._generate_with_provider(
                provider, model, messages, temperature, max_tokens, **kwargs
            )
        except Exception as primary_error:
            self.logger.warning(
                "Primary model failed, attempting fallback",
                provider=provider,
                model=model,
                error=str(primary_error)
            )

            # Try fallback if enabled
            if self.settings.auto_fallback_enabled:
                return await self._generate_with_fallback(
                    messages, temperature, max_tokens, task_type, primary_error, **kwargs
                )
            else:
                raise ExternalServiceError(f"LLM generation failed: {str(primary_error)}")

    async def _generate_with_provider(
        self,
        provider: LLMProvider,
        model: str,
        messages: List[LLMMessage],
        temperature: float,
        max_tokens: Optional[int],
        **kwargs
    ) -> LLMResponse:
        """Generate response with specific provider."""
        if provider == LLMProvider.OPENAI:
            return await self._generate_openai_response(
                messages, model, temperature, max_tokens, **kwargs
            )
        elif provider == LLMProvider.ANTHROPIC:
            return await self._generate_anthropic_response(
                messages, model, temperature, max_tokens, **kwargs
            )
        elif provider == LLMProvider.GOOGLE:
            return await self._generate_google_response(
                messages, model, temperature, max_tokens, **kwargs
            )
        else:
            raise ExternalServiceError(f"Unsupported LLM provider: {provider}")

    async def _generate_with_fallback(
        self,
        messages: List[LLMMessage],
        temperature: float,
        max_tokens: Optional[int],
        task_type: str,
        primary_error: Exception,
        **kwargs
    ) -> LLMResponse:
        """Try fallback providers and models."""
        fallback_attempts = []

        # Try fallback provider
        fallback_provider = self._get_fallback_provider()
        if fallback_provider:
            try:
                fallback_model = self._get_optimal_model(fallback_provider, task_type)
                response = await self._generate_with_provider(
                    fallback_provider, fallback_model, messages, temperature, max_tokens, **kwargs
                )

                # Mark as fallback response
                response.metadata["fallback_used"] = True
                response.metadata["primary_error"] = str(primary_error)

                self.logger.info(
                    "Fallback successful",
                    fallback_provider=fallback_provider,
                    fallback_model=fallback_model
                )

                return response

            except Exception as fallback_error:
                fallback_attempts.append({
                    "provider": fallback_provider,
                    "error": str(fallback_error)
                })

        # If all fallbacks fail, raise comprehensive error
        raise ExternalServiceError(
            f"All LLM providers failed. Primary: {str(primary_error)}. "
            f"Fallbacks: {fallback_attempts}"
        )
    
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
    
    async def _generate_google_response(
        self,
        messages: List[LLMMessage],
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        **kwargs
    ) -> LLMResponse:
        """Generate response using Google Gemini API (placeholder for 2025)."""
        # TODO: Implement Google Gemini API integration
        raise ExternalServiceError("Google Gemini integration not yet implemented")

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
    
    def _get_optimal_provider(self, task_type: str = "general") -> LLMProvider:
        """Get optimal LLM provider based on task type and availability (2025)."""
        # Task-specific provider preferences
        task_preferences = {
            "code": LLMProvider.OPENAI,  # GPT-4.1 excels at coding
            "reasoning": LLMProvider.ANTHROPIC,  # Claude 3.7 excels at reasoning
            "analysis": LLMProvider.ANTHROPIC,  # Claude 3.7 excels at analysis
            "creative": LLMProvider.ANTHROPIC,  # Claude 3.7 excels at creative tasks
            "general": LLMProvider.OPENAI,  # GPT-4.1 for general tasks
            "vision": LLMProvider.OPENAI,  # GPT-4o for vision tasks
        }

        preferred_provider = task_preferences.get(task_type, LLMProvider.OPENAI)

        # Check if preferred provider is available
        if preferred_provider == LLMProvider.OPENAI and self._openai_client:
            return LLMProvider.OPENAI
        elif preferred_provider == LLMProvider.ANTHROPIC and self._anthropic_client:
            return LLMProvider.ANTHROPIC

        # Fallback to primary provider setting
        primary = getattr(self.settings, 'primary_llm_provider', 'openai')
        if primary == "openai" and self._openai_client:
            return LLMProvider.OPENAI
        elif primary == "anthropic" and self._anthropic_client:
            return LLMProvider.ANTHROPIC

        # Final fallback to any available provider
        if self._openai_client:
            return LLMProvider.OPENAI
        elif self._anthropic_client:
            return LLMProvider.ANTHROPIC
        else:
            raise ExternalServiceError("No LLM providers configured. Please add OpenAI or Anthropic API keys.")

    def _get_optimal_model(self, provider: LLMProvider, task_type: str = "general") -> str:
        """Get optimal model for provider and task type (2025)."""
        if provider == LLMProvider.OPENAI:
            # Task-specific model selection for OpenAI
            if task_type == "code":
                return self.settings.openai_code_model  # gpt-4.1
            elif task_type == "vision":
                return self.settings.openai_vision_model  # gpt-4o
            else:
                return self.settings.openai_model  # gpt-4.1

        elif provider == LLMProvider.ANTHROPIC:
            return self.settings.anthropic_model  # claude-3-7-sonnet-latest

        elif provider == LLMProvider.GOOGLE:
            return self.settings.gemini_model  # gemini-2.5-pro

        else:
            return "unknown-model"

    def _get_fallback_provider(self) -> Optional[LLMProvider]:
        """Get fallback provider."""
        fallback = getattr(self.settings, 'fallback_llm_provider', 'anthropic')

        if fallback == "anthropic" and self._anthropic_client:
            return LLMProvider.ANTHROPIC
        elif fallback == "openai" and self._openai_client:
            return LLMProvider.OPENAI
        elif fallback == "google" and self._google_client:
            return LLMProvider.GOOGLE

        # Try any available provider as last resort
        if self._anthropic_client:
            return LLMProvider.ANTHROPIC
        elif self._openai_client:
            return LLMProvider.OPENAI

        return None
    
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
        """Get comprehensive status of all LLM providers (2025)."""
        return {
            "openai": {
                "available": self._openai_client is not None,
                "healthy": self._model_health.get("openai", False),
                "primary_model": self.settings.openai_model if self._openai_client else None,
                "fallback_model": self.settings.openai_fallback_model if self._openai_client else None,
                "specialized_models": {
                    "code": self.settings.openai_code_model,
                    "vision": self.settings.openai_vision_model,
                    "embedding": self.settings.openai_embedding_model
                } if self._openai_client else None
            },
            "anthropic": {
                "available": self._anthropic_client is not None,
                "healthy": self._model_health.get("anthropic", False),
                "primary_model": self.settings.anthropic_model if self._anthropic_client else None,
                "fallback_model": self.settings.anthropic_fallback_model if self._anthropic_client else None
            },
            "google": {
                "available": self._google_client is not None,
                "healthy": self._model_health.get("google", False),
                "model": self.settings.gemini_model if self._google_client else None
            },
            "configuration": {
                "primary_provider": self.settings.primary_llm_provider,
                "fallback_provider": self.settings.fallback_llm_provider,
                "auto_fallback_enabled": self.settings.auto_fallback_enabled
            },
            "performance": {
                "fallback_attempts": self._fallback_attempts
            }
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all available providers."""
        health_results = {}

        # Test OpenAI
        if self._openai_client:
            try:
                test_messages = [LLMMessage("user", "Hello")]
                await self._generate_openai_response(
                    test_messages, self.settings.openai_model, 0.1, 10
                )
                health_results["openai"] = {"status": "healthy", "latency_ms": 0}
                self._model_health["openai"] = True
            except Exception as e:
                health_results["openai"] = {"status": "unhealthy", "error": str(e)}
                self._model_health["openai"] = False

        # Test Anthropic
        if self._anthropic_client:
            try:
                test_messages = [LLMMessage("user", "Hello")]
                await self._generate_anthropic_response(
                    test_messages, self.settings.anthropic_model, 0.1, 10
                )
                health_results["anthropic"] = {"status": "healthy", "latency_ms": 0}
                self._model_health["anthropic"] = True
            except Exception as e:
                health_results["anthropic"] = {"status": "unhealthy", "error": str(e)}
                self._model_health["anthropic"] = False

        return health_results
