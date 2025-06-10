"""OpenAI provider plugin for the LLM contract framework.

This module provides integration with OpenAI's API through the plugin system,
including GPT models, function calling, and streaming support.
"""

import os
import asyncio
from typing import Any, Dict, List, Optional, AsyncIterable
import logging

try:
    import openai
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    # Mock classes for when OpenAI is not available
    class AsyncOpenAI:
        pass

from .provider_plugin import (
    ProviderPlugin, ProviderType, ProviderCapabilities, ModelCapability,
    UnifiedRequest, UnifiedResponse
)

logger = logging.getLogger(__name__)


class OpenAIPlugin(ProviderPlugin):
    """OpenAI provider plugin implementation."""
    
    def __init__(self,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 organization: Optional[str] = None,
                 default_model: str = "gpt-3.5-turbo",
                 contract_enabled: bool = True,
                 timeout: float = 30.0,
                 max_retries: int = 3):
        """Initialize OpenAI plugin.
        
        Args:
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)
            base_url: Custom base URL for API
            organization: OpenAI organization ID
            default_model: Default model to use
            contract_enabled: Whether contract enforcement is enabled
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package is required for OpenAI plugin")
        
        super().__init__(
            provider_type=ProviderType.OPENAI,
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url,
            default_model=default_model,
            contract_enabled=contract_enabled
        )
        
        self.organization = organization
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Initialize OpenAI client
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            organization=self.organization,
            timeout=self.timeout,
            max_retries=self.max_retries
        )
        
        # OpenAI-specific metrics
        self.openai_metrics = {
            "tokens_used": 0,
            "cost_estimate": 0.0,
            "function_calls": 0,
            "tool_calls": 0
        }
    
    def get_capabilities(self) -> ProviderCapabilities:
        """Get OpenAI provider capabilities."""
        return ProviderCapabilities(
            provider_type=ProviderType.OPENAI,
            supported_models=[
                "gpt-4", "gpt-4-turbo", "gpt-4-turbo-preview",
                "gpt-4-0125-preview", "gpt-4-1106-preview",
                "gpt-3.5-turbo", "gpt-3.5-turbo-0125", "gpt-3.5-turbo-1106",
                "gpt-3.5-turbo-16k", "gpt-3.5-turbo-instruct",
                "text-davinci-003", "text-davinci-002",
                "code-davinci-002", "text-ada-001", "text-babbage-001", "text-curie-001"
            ],
            model_capabilities={
                "gpt-4": [ModelCapability.CHAT_COMPLETION, ModelCapability.FUNCTION_CALLING, 
                         ModelCapability.STREAMING, ModelCapability.VISION, ModelCapability.REASONING],
                "gpt-4-turbo": [ModelCapability.CHAT_COMPLETION, ModelCapability.FUNCTION_CALLING,
                               ModelCapability.STREAMING, ModelCapability.VISION, ModelCapability.REASONING],
                "gpt-3.5-turbo": [ModelCapability.CHAT_COMPLETION, ModelCapability.FUNCTION_CALLING,
                                 ModelCapability.STREAMING, ModelCapability.CODE_GENERATION],
                "text-davinci-003": [ModelCapability.TEXT_GENERATION, ModelCapability.CODE_GENERATION],
            },
            max_tokens={
                "gpt-4": 8192,
                "gpt-4-turbo": 4096,
                "gpt-4-turbo-preview": 4096,
                "gpt-3.5-turbo": 4096,
                "gpt-3.5-turbo-16k": 16384,
                "text-davinci-003": 4097,
            },
            context_windows={
                "gpt-4": 8192,
                "gpt-4-turbo": 128000,
                "gpt-4-turbo-preview": 128000,
                "gpt-3.5-turbo": 4096,
                "gpt-3.5-turbo-16k": 16384,
                "text-davinci-003": 4097,
            },
            pricing={
                "gpt-4": {"input_per_token": 0.00003, "output_per_token": 0.00006},
                "gpt-4-turbo": {"input_per_token": 0.00001, "output_per_token": 0.00003},
                "gpt-3.5-turbo": {"input_per_token": 0.0000005, "output_per_token": 0.0000015},
                "text-davinci-003": {"input_per_token": 0.00002, "output_per_token": 0.00002},
            },
            rate_limits={
                "gpt-4": {"requests_per_minute": 200, "tokens_per_minute": 40000},
                "gpt-3.5-turbo": {"requests_per_minute": 3500, "tokens_per_minute": 90000},
            },
            supports_streaming=True,
            supports_async=True,
            api_version="v1"
        )
    
    async def complete_async(self, request: UnifiedRequest) -> UnifiedResponse:
        """Complete a chat/text generation request asynchronously."""
        try:
            # Convert to OpenAI format
            openai_request = request.to_provider_format(ProviderType.OPENAI)
            
            # Use appropriate model if not specified
            if not openai_request.get("model"):
                openai_request["model"] = self.default_model
            
            # Determine if this is a chat completion or text completion
            if "messages" in openai_request:
                response = await self.client.chat.completions.create(**openai_request)
                
                # Convert response
                choice = response.choices[0]
                message = choice.message
                
                unified_response = UnifiedResponse(
                    content=message.content or "",
                    model=response.model,
                    usage={
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    },
                    finish_reason=choice.finish_reason,
                    function_call=message.function_call.model_dump() if message.function_call else None,
                    tool_calls=[tc.model_dump() for tc in message.tool_calls] if message.tool_calls else None,
                    response_id=response.id,
                    created=response.created,
                    system_fingerprint=response.system_fingerprint
                )
                
                # Update metrics
                self._update_metrics(response.usage, message.function_call, message.tool_calls)
                
                return unified_response
            
            else:
                # Text completion (legacy)
                response = await self.client.completions.create(**openai_request)
                
                choice = response.choices[0]
                
                unified_response = UnifiedResponse(
                    content=choice.text,
                    model=response.model,
                    usage={
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    },
                    finish_reason=choice.finish_reason,
                    response_id=response.id,
                    created=response.created
                )
                
                # Update metrics
                self._update_metrics(response.usage)
                
                return unified_response
                
        except Exception as e:
            logger.error(f"OpenAI API request failed: {e}")
            raise
    
    async def complete_stream_async(self, request: UnifiedRequest) -> AsyncIterable[UnifiedResponse]:
        """Complete a streaming chat/text generation request asynchronously."""
        try:
            # Convert to OpenAI format and enable streaming
            openai_request = request.to_provider_format(ProviderType.OPENAI)
            openai_request["stream"] = True
            
            if not openai_request.get("model"):
                openai_request["model"] = self.default_model
            
            # Use chat completions for streaming
            if "messages" in openai_request:
                stream = await self.client.chat.completions.create(**openai_request)
                
                accumulated_content = ""
                
                async for chunk in stream:
                    if chunk.choices:
                        choice = chunk.choices[0]
                        delta = choice.delta
                        
                        content = delta.content or ""
                        accumulated_content += content
                        
                        unified_response = UnifiedResponse(
                            content=content,
                            model=chunk.model,
                            finish_reason=choice.finish_reason,
                            response_id=chunk.id,
                            created=chunk.created,
                            system_fingerprint=chunk.system_fingerprint,
                            metadata={"accumulated_content": accumulated_content}
                        )
                        
                        # Handle function/tool calls in streaming
                        if delta.function_call:
                            unified_response.function_call = delta.function_call.model_dump()
                            self.openai_metrics["function_calls"] += 1
                        
                        if delta.tool_calls:
                            unified_response.tool_calls = [tc.model_dump() for tc in delta.tool_calls]
                            self.openai_metrics["tool_calls"] += len(delta.tool_calls)
                        
                        yield unified_response
            
            else:
                # Text completion streaming (legacy)
                stream = await self.client.completions.create(**openai_request)
                
                accumulated_content = ""
                
                async for chunk in stream:
                    if chunk.choices:
                        choice = chunk.choices[0]
                        content = choice.text or ""
                        accumulated_content += content
                        
                        unified_response = UnifiedResponse(
                            content=content,
                            model=chunk.model,
                            finish_reason=choice.finish_reason,
                            response_id=chunk.id,
                            created=chunk.created,
                            metadata={"accumulated_content": accumulated_content}
                        )
                        
                        yield unified_response
                        
        except Exception as e:
            logger.error(f"OpenAI streaming request failed: {e}")
            raise
    
    def _update_metrics(self, usage, function_call=None, tool_calls=None):
        """Update OpenAI-specific metrics."""
        if usage:
            self.openai_metrics["tokens_used"] += usage.total_tokens
            
            # Estimate cost (this is a rough estimate)
            model = self.default_model
            capabilities = self.get_capabilities()
            pricing = capabilities.pricing.get(model, {"input_per_token": 0, "output_per_token": 0})
            
            estimated_cost = (usage.prompt_tokens * pricing["input_per_token"] + 
                            usage.completion_tokens * pricing["output_per_token"])
            self.openai_metrics["cost_estimate"] += estimated_cost
        
        if function_call:
            self.openai_metrics["function_calls"] += 1
        
        if tool_calls:
            self.openai_metrics["tool_calls"] += len(tool_calls)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get plugin metrics including OpenAI-specific metrics."""
        base_metrics = super().get_metrics()
        base_metrics.update({
            "openai_metrics": self.openai_metrics,
            "provider_specific": {
                "api_version": "v1",
                "client_info": {
                    "base_url": str(self.client.base_url) if self.client.base_url else None,
                    "timeout": self.timeout,
                    "max_retries": self.max_retries
                }
            }
        })
        return base_metrics
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on OpenAI API."""
        health_status = super().health_check()
        
        try:
            # Try to make a simple API call to check connectivity
            # This is a simple check - in practice you might want to cache this
            # or use a dedicated health endpoint if available
            
            health_status.update({
                "openai_specific": {
                    "api_key_configured": bool(self.api_key),
                    "client_initialized": self.client is not None,
                    "base_url": str(self.client.base_url) if self.client and self.client.base_url else None
                }
            })
            
        except Exception as e:
            health_status["status"] = "error"
            health_status["error"] = f"OpenAI health check failed: {e}"
        
        return health_status
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models from OpenAI API."""
        try:
            models = await self.client.models.list()
            return [model.model_dump() for model in models.data]
        except Exception as e:
            logger.error(f"Failed to list OpenAI models: {e}")
            return []
    
    async def get_model_info(self, model: str) -> Dict[str, Any]:
        """Get information about a specific model."""
        try:
            model_info = await self.client.models.retrieve(model)
            return model_info.model_dump()
        except Exception as e:
            logger.error(f"Failed to get info for model {model}: {e}")
            return {}
    
    def cleanup(self):
        """Cleanup resources."""
        if hasattr(self.client, 'close'):
            asyncio.create_task(self.client.close())
        logger.info("OpenAI plugin cleanup completed")