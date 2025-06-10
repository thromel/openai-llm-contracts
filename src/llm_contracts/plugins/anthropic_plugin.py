"""Anthropic Claude provider plugin for the LLM contract framework.

This module provides integration with Anthropic's Claude API through the plugin system,
including Claude models and streaming support.
"""

import os
import asyncio
from typing import Any, Dict, List, Optional, AsyncIterable
import logging

try:
    import anthropic
    from anthropic import AsyncAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    # Mock classes for when Anthropic is not available
    class AsyncAnthropic:
        pass

from .provider_plugin import (
    ProviderPlugin, ProviderType, ProviderCapabilities, ModelCapability,
    UnifiedRequest, UnifiedResponse
)

logger = logging.getLogger(__name__)


class AnthropicPlugin(ProviderPlugin):
    """Anthropic Claude provider plugin implementation."""
    
    def __init__(self,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 default_model: str = "claude-3-sonnet-20240229",
                 contract_enabled: bool = True,
                 timeout: float = 60.0,
                 max_retries: int = 3):
        """Initialize Anthropic plugin.
        
        Args:
            api_key: Anthropic API key (uses ANTHROPIC_API_KEY env var if not provided)
            base_url: Custom base URL for API
            default_model: Default model to use
            contract_enabled: Whether contract enforcement is enabled
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
        """
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic package is required for Anthropic plugin")
        
        super().__init__(
            provider_type=ProviderType.ANTHROPIC,
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
            base_url=base_url,
            default_model=default_model,
            contract_enabled=contract_enabled
        )
        
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Initialize Anthropic client
        self.client = AsyncAnthropic(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries
        )
        
        # Anthropic-specific metrics
        self.anthropic_metrics = {
            "tokens_used": 0,
            "cost_estimate": 0.0,
            "reasoning_steps": 0,
            "thinking_tokens": 0
        }
    
    def get_capabilities(self) -> ProviderCapabilities:
        """Get Anthropic provider capabilities."""
        return ProviderCapabilities(
            provider_type=ProviderType.ANTHROPIC,
            supported_models=[
                "claude-3-5-sonnet-20241022",
                "claude-3-5-sonnet-20240620", 
                "claude-3-sonnet-20240229",
                "claude-3-opus-20240229",
                "claude-3-haiku-20240307",
                "claude-2.1",
                "claude-2.0",
                "claude-instant-1.2"
            ],
            model_capabilities={
                "claude-3-5-sonnet-20241022": [ModelCapability.CHAT_COMPLETION, ModelCapability.STREAMING,
                                               ModelCapability.VISION, ModelCapability.REASONING,
                                               ModelCapability.CODE_GENERATION],
                "claude-3-5-sonnet-20240620": [ModelCapability.CHAT_COMPLETION, ModelCapability.STREAMING,
                                               ModelCapability.VISION, ModelCapability.CODE_GENERATION],
                "claude-3-sonnet-20240229": [ModelCapability.CHAT_COMPLETION, ModelCapability.STREAMING,
                                            ModelCapability.VISION, ModelCapability.CODE_GENERATION],
                "claude-3-opus-20240229": [ModelCapability.CHAT_COMPLETION, ModelCapability.STREAMING,
                                          ModelCapability.VISION, ModelCapability.REASONING],
                "claude-3-haiku-20240307": [ModelCapability.CHAT_COMPLETION, ModelCapability.STREAMING,
                                           ModelCapability.VISION],
                "claude-2.1": [ModelCapability.CHAT_COMPLETION, ModelCapability.STREAMING],
                "claude-2.0": [ModelCapability.CHAT_COMPLETION, ModelCapability.STREAMING],
                "claude-instant-1.2": [ModelCapability.CHAT_COMPLETION, ModelCapability.STREAMING],
            },
            max_tokens={
                "claude-3-5-sonnet-20241022": 8192,
                "claude-3-5-sonnet-20240620": 8192,
                "claude-3-sonnet-20240229": 4096,
                "claude-3-opus-20240229": 4096,
                "claude-3-haiku-20240307": 4096,
                "claude-2.1": 4096,
                "claude-2.0": 4096,
                "claude-instant-1.2": 4096,
            },
            context_windows={
                "claude-3-5-sonnet-20241022": 200000,
                "claude-3-5-sonnet-20240620": 200000,
                "claude-3-sonnet-20240229": 200000,
                "claude-3-opus-20240229": 200000,
                "claude-3-haiku-20240307": 200000,
                "claude-2.1": 200000,
                "claude-2.0": 100000,
                "claude-instant-1.2": 100000,
            },
            pricing={
                "claude-3-5-sonnet-20241022": {"input_per_token": 0.000003, "output_per_token": 0.000015},
                "claude-3-5-sonnet-20240620": {"input_per_token": 0.000003, "output_per_token": 0.000015},
                "claude-3-sonnet-20240229": {"input_per_token": 0.000003, "output_per_token": 0.000015},
                "claude-3-opus-20240229": {"input_per_token": 0.000015, "output_per_token": 0.000075},
                "claude-3-haiku-20240307": {"input_per_token": 0.00000025, "output_per_token": 0.00000125},
                "claude-2.1": {"input_per_token": 0.000008, "output_per_token": 0.000024},
                "claude-instant-1.2": {"input_per_token": 0.00000163, "output_per_token": 0.00000551},
            },
            rate_limits={
                "claude-3-5-sonnet-20241022": {"requests_per_minute": 1000, "tokens_per_minute": 200000},
                "claude-3-sonnet-20240229": {"requests_per_minute": 1000, "tokens_per_minute": 200000},
                "claude-3-opus-20240229": {"requests_per_minute": 1000, "tokens_per_minute": 80000},
                "claude-3-haiku-20240307": {"requests_per_minute": 1000, "tokens_per_minute": 200000},
            },
            supports_streaming=True,
            supports_async=True,
            api_version="2023-06-01"
        )
    
    async def complete_async(self, request: UnifiedRequest) -> UnifiedResponse:
        """Complete a chat/text generation request asynchronously."""
        try:
            # Convert to Anthropic format
            anthropic_request = request.to_provider_format(ProviderType.ANTHROPIC)
            
            # Use appropriate model if not specified
            if not anthropic_request.get("model"):
                anthropic_request["model"] = self.default_model
            
            # Make the API call
            response = await self.client.messages.create(**anthropic_request)
            
            # Convert response to unified format
            unified_response = UnifiedResponse.from_anthropic_format(response.model_dump())
            
            # Update metrics
            self._update_metrics(response.usage)
            
            return unified_response
                
        except Exception as e:
            logger.error(f"Anthropic API request failed: {e}")
            raise
    
    async def complete_stream_async(self, request: UnifiedRequest) -> AsyncIterable[UnifiedResponse]:
        """Complete a streaming chat/text generation request asynchronously."""
        try:
            # Convert to Anthropic format and enable streaming
            anthropic_request = request.to_provider_format(ProviderType.ANTHROPIC)
            anthropic_request["stream"] = True
            
            if not anthropic_request.get("model"):
                anthropic_request["model"] = self.default_model
            
            # Start streaming
            accumulated_content = ""
            
            async with self.client.messages.stream(**anthropic_request) as stream:
                async for event in stream:
                    if event.type == "content_block_delta":
                        if hasattr(event.delta, 'text'):
                            content = event.delta.text
                            accumulated_content += content
                            
                            unified_response = UnifiedResponse(
                                content=content,
                                model=anthropic_request["model"],
                                metadata={"accumulated_content": accumulated_content}
                            )
                            
                            yield unified_response
                    
                    elif event.type == "message_delta":
                        if hasattr(event, 'delta') and hasattr(event.delta, 'stop_reason'):
                            # Final chunk with stop reason
                            unified_response = UnifiedResponse(
                                content="",
                                model=anthropic_request["model"],
                                finish_reason=event.delta.stop_reason,
                                metadata={"accumulated_content": accumulated_content}
                            )
                            
                            yield unified_response
                    
                    elif event.type == "message_stop":
                        # Stream completed
                        break
                        
        except Exception as e:
            logger.error(f"Anthropic streaming request failed: {e}")
            raise
    
    def _update_metrics(self, usage):
        """Update Anthropic-specific metrics."""
        if usage:
            input_tokens = getattr(usage, 'input_tokens', 0)
            output_tokens = getattr(usage, 'output_tokens', 0)
            total_tokens = input_tokens + output_tokens
            
            self.anthropic_metrics["tokens_used"] += total_tokens
            
            # Estimate cost
            model = self.default_model
            capabilities = self.get_capabilities()
            pricing = capabilities.pricing.get(model, {"input_per_token": 0, "output_per_token": 0})
            
            estimated_cost = (input_tokens * pricing["input_per_token"] + 
                            output_tokens * pricing["output_per_token"])
            self.anthropic_metrics["cost_estimate"] += estimated_cost
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get plugin metrics including Anthropic-specific metrics."""
        base_metrics = super().get_metrics()
        base_metrics.update({
            "anthropic_metrics": self.anthropic_metrics,
            "provider_specific": {
                "api_version": "2023-06-01",
                "client_info": {
                    "base_url": str(self.client.base_url) if self.client.base_url else None,
                    "timeout": self.timeout,
                    "max_retries": self.max_retries
                }
            }
        })
        return base_metrics
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on Anthropic API."""
        health_status = super().health_check()
        
        try:
            health_status.update({
                "anthropic_specific": {
                    "api_key_configured": bool(self.api_key),
                    "client_initialized": self.client is not None,
                    "base_url": str(self.client.base_url) if self.client and self.client.base_url else None
                }
            })
            
        except Exception as e:
            health_status["status"] = "error"
            health_status["error"] = f"Anthropic health check failed: {e}"
        
        return health_status
    
    def cleanup(self):
        """Cleanup resources."""
        if hasattr(self.client, 'close'):
            asyncio.create_task(self.client.close())
        logger.info("Anthropic plugin cleanup completed")