"""Azure OpenAI provider plugin for the LLM contract framework.

This module provides integration with Azure OpenAI Service through the plugin system.
"""

import os
import asyncio
from typing import Any, Dict, List, Optional, AsyncIterable
import logging

try:
    from openai import AsyncAzureOpenAI
    AZURE_OPENAI_AVAILABLE = True
except ImportError:
    AZURE_OPENAI_AVAILABLE = False
    # Mock class for when Azure OpenAI is not available
    class AsyncAzureOpenAI:
        pass

from .provider_plugin import (
    ProviderPlugin, ProviderType, ProviderCapabilities, ModelCapability,
    UnifiedRequest, UnifiedResponse
)

logger = logging.getLogger(__name__)


class AzureOpenAIPlugin(ProviderPlugin):
    """Azure OpenAI provider plugin implementation."""
    
    def __init__(self,
                 api_key: Optional[str] = None,
                 azure_endpoint: Optional[str] = None,
                 api_version: str = "2024-02-15-preview",
                 default_model: str = "gpt-35-turbo",
                 contract_enabled: bool = True,
                 timeout: float = 30.0,
                 max_retries: int = 3):
        """Initialize Azure OpenAI plugin.
        
        Args:
            api_key: Azure OpenAI API key (uses AZURE_OPENAI_API_KEY env var if not provided)
            azure_endpoint: Azure OpenAI endpoint (uses AZURE_OPENAI_ENDPOINT env var if not provided)
            api_version: Azure OpenAI API version
            default_model: Default model deployment name to use
            contract_enabled: Whether contract enforcement is enabled
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
        """
        if not AZURE_OPENAI_AVAILABLE:
            raise ImportError("OpenAI package with Azure support is required for Azure OpenAI plugin")
        
        super().__init__(
            provider_type=ProviderType.AZURE_OPENAI,
            api_key=api_key or os.getenv("AZURE_OPENAI_API_KEY"),
            base_url=azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT"),
            default_model=default_model,
            contract_enabled=contract_enabled
        )
        
        self.azure_endpoint = self.base_url
        self.api_version = api_version
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Initialize Azure OpenAI client
        self.client = AsyncAzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.azure_endpoint,
            api_version=self.api_version,
            timeout=self.timeout,
            max_retries=self.max_retries
        )
        
        # Azure-specific metrics
        self.azure_metrics = {
            "tokens_used": 0,
            "cost_estimate": 0.0,
            "deployments_used": set(),
            "function_calls": 0,
            "tool_calls": 0
        }
    
    def get_capabilities(self) -> ProviderCapabilities:
        """Get Azure OpenAI provider capabilities."""
        return ProviderCapabilities(
            provider_type=ProviderType.AZURE_OPENAI,
            supported_models=[
                # Common Azure OpenAI deployment names (these are configurable in Azure)
                "gpt-4", "gpt-4-turbo", "gpt-4-32k",
                "gpt-35-turbo", "gpt-35-turbo-16k",
                "text-embedding-ada-002",
                "text-davinci-003", "text-davinci-002",
                "code-davinci-002"
            ],
            model_capabilities={
                "gpt-4": [ModelCapability.CHAT_COMPLETION, ModelCapability.FUNCTION_CALLING,
                         ModelCapability.STREAMING, ModelCapability.REASONING],
                "gpt-4-turbo": [ModelCapability.CHAT_COMPLETION, ModelCapability.FUNCTION_CALLING,
                               ModelCapability.STREAMING, ModelCapability.VISION, ModelCapability.REASONING],
                "gpt-35-turbo": [ModelCapability.CHAT_COMPLETION, ModelCapability.FUNCTION_CALLING,
                                ModelCapability.STREAMING, ModelCapability.CODE_GENERATION],
                "text-davinci-003": [ModelCapability.TEXT_GENERATION, ModelCapability.CODE_GENERATION],
                "text-embedding-ada-002": [ModelCapability.EMBEDDINGS],
            },
            max_tokens={
                "gpt-4": 8192,
                "gpt-4-turbo": 4096,
                "gpt-4-32k": 32768,
                "gpt-35-turbo": 4096,
                "gpt-35-turbo-16k": 16384,
                "text-davinci-003": 4097,
            },
            context_windows={
                "gpt-4": 8192,
                "gpt-4-turbo": 128000,
                "gpt-4-32k": 32768,
                "gpt-35-turbo": 4096,
                "gpt-35-turbo-16k": 16384,
                "text-davinci-003": 4097,
            },
            pricing={
                # Azure pricing varies by region and commitment
                # These are example values - actual pricing should be configured
                "gpt-4": {"input_per_token": 0.00003, "output_per_token": 0.00006},
                "gpt-4-turbo": {"input_per_token": 0.00001, "output_per_token": 0.00003},
                "gpt-35-turbo": {"input_per_token": 0.0000005, "output_per_token": 0.0000015},
            },
            rate_limits={
                # Azure rate limits are configurable per deployment
                "gpt-4": {"requests_per_minute": 200, "tokens_per_minute": 40000},
                "gpt-35-turbo": {"requests_per_minute": 3500, "tokens_per_minute": 90000},
            },
            supports_streaming=True,
            supports_async=True,
            api_version=self.api_version
        )
    
    async def complete_async(self, request: UnifiedRequest) -> UnifiedResponse:
        """Complete a chat/text generation request asynchronously."""
        try:
            # Convert to OpenAI format (Azure OpenAI uses same format)
            azure_request = request.to_provider_format(ProviderType.OPENAI)
            
            # Use deployment name as model
            deployment_name = azure_request.get("model", self.default_model)
            azure_request["model"] = deployment_name
            
            # Track deployment usage
            self.azure_metrics["deployments_used"].add(deployment_name)
            
            # Determine if this is a chat completion or text completion
            if "messages" in azure_request:
                response = await self.client.chat.completions.create(**azure_request)
                
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
                # Text completion
                response = await self.client.completions.create(**azure_request)
                
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
            logger.error(f"Azure OpenAI API request failed: {e}")
            raise
    
    async def complete_stream_async(self, request: UnifiedRequest) -> AsyncIterable[UnifiedResponse]:
        """Complete a streaming chat/text generation request asynchronously."""
        try:
            # Convert to Azure OpenAI format and enable streaming
            azure_request = request.to_provider_format(ProviderType.OPENAI)
            azure_request["stream"] = True
            
            deployment_name = azure_request.get("model", self.default_model)
            azure_request["model"] = deployment_name
            
            # Track deployment usage
            self.azure_metrics["deployments_used"].add(deployment_name)
            
            # Use chat completions for streaming
            if "messages" in azure_request:
                stream = await self.client.chat.completions.create(**azure_request)
                
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
                            self.azure_metrics["function_calls"] += 1
                        
                        if delta.tool_calls:
                            unified_response.tool_calls = [tc.model_dump() for tc in delta.tool_calls]
                            self.azure_metrics["tool_calls"] += len(delta.tool_calls)
                        
                        yield unified_response
            
            else:
                # Text completion streaming
                stream = await self.client.completions.create(**azure_request)
                
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
            logger.error(f"Azure OpenAI streaming request failed: {e}")
            raise
    
    def _update_metrics(self, usage, function_call=None, tool_calls=None):
        """Update Azure-specific metrics."""
        if usage:
            self.azure_metrics["tokens_used"] += usage.total_tokens
            
            # Estimate cost (this would need to be configured per Azure pricing)
            # This is a placeholder calculation
            deployment = self.default_model
            capabilities = self.get_capabilities()
            pricing = capabilities.pricing.get(deployment, {"input_per_token": 0, "output_per_token": 0})
            
            estimated_cost = (usage.prompt_tokens * pricing["input_per_token"] + 
                            usage.completion_tokens * pricing["output_per_token"])
            self.azure_metrics["cost_estimate"] += estimated_cost
        
        if function_call:
            self.azure_metrics["function_calls"] += 1
        
        if tool_calls:
            self.azure_metrics["tool_calls"] += len(tool_calls)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get plugin metrics including Azure-specific metrics."""
        base_metrics = super().get_metrics()
        
        # Convert set to list for JSON serialization
        azure_metrics_copy = self.azure_metrics.copy()
        azure_metrics_copy["deployments_used"] = list(azure_metrics_copy["deployments_used"])
        
        base_metrics.update({
            "azure_metrics": azure_metrics_copy,
            "provider_specific": {
                "api_version": self.api_version,
                "client_info": {
                    "azure_endpoint": self.azure_endpoint,
                    "timeout": self.timeout,
                    "max_retries": self.max_retries
                }
            }
        })
        return base_metrics
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on Azure OpenAI API."""
        health_status = super().health_check()
        
        try:
            health_status.update({
                "azure_specific": {
                    "api_key_configured": bool(self.api_key),
                    "azure_endpoint_configured": bool(self.azure_endpoint),
                    "client_initialized": self.client is not None,
                    "api_version": self.api_version,
                    "azure_endpoint": self.azure_endpoint
                }
            })
            
        except Exception as e:
            health_status["status"] = "error"
            health_status["error"] = f"Azure OpenAI health check failed: {e}"
        
        return health_status
    
    async def list_deployments(self) -> List[Dict[str, Any]]:
        """List available deployments from Azure OpenAI."""
        try:
            # Azure OpenAI API doesn't have a direct deployments endpoint
            # This would typically be done through Azure Resource Manager APIs
            # For now, return the supported models as potential deployments
            capabilities = self.get_capabilities()
            deployments = []
            
            for model in capabilities.supported_models:
                deployments.append({
                    "deployment_name": model,
                    "model_name": model,
                    "capabilities": capabilities.model_capabilities.get(model, []),
                    "max_tokens": capabilities.max_tokens.get(model, 4096),
                    "context_window": capabilities.context_windows.get(model, 4096)
                })
            
            return deployments
            
        except Exception as e:
            logger.error(f"Failed to list Azure OpenAI deployments: {e}")
            return []
    
    def cleanup(self):
        """Cleanup resources."""
        if hasattr(self.client, 'close'):
            asyncio.create_task(self.client.close())
        logger.info("Azure OpenAI plugin cleanup completed")