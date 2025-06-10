"""Google AI provider plugin for the LLM contract framework.

This module provides integration with Google's Gemini API through the plugin system,
including Gemini models and streaming support.
"""

import os
import asyncio
from typing import Any, Dict, List, Optional, AsyncIterable
import logging

try:
    import google.generativeai as genai
    from google.generativeai.types import GenerateContentResponse
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False

from .provider_plugin import (
    ProviderPlugin, ProviderType, ProviderCapabilities, ModelCapability,
    UnifiedRequest, UnifiedResponse
)

logger = logging.getLogger(__name__)


class GooglePlugin(ProviderPlugin):
    """Google AI provider plugin implementation."""
    
    def __init__(self,
                 api_key: Optional[str] = None,
                 default_model: str = "gemini-1.5-pro",
                 contract_enabled: bool = True,
                 timeout: float = 60.0):
        """Initialize Google AI plugin.
        
        Args:
            api_key: Google AI API key (uses GOOGLE_API_KEY env var if not provided)
            default_model: Default model to use
            contract_enabled: Whether contract enforcement is enabled
            timeout: Request timeout in seconds
        """
        if not GOOGLE_AI_AVAILABLE:
            raise ImportError("google-generativeai package is required for Google plugin")
        
        super().__init__(
            provider_type=ProviderType.GOOGLE,
            api_key=api_key or os.getenv("GOOGLE_API_KEY"),
            default_model=default_model,
            contract_enabled=contract_enabled
        )
        
        self.timeout = timeout
        
        # Configure Google AI
        if self.api_key:
            genai.configure(api_key=self.api_key)
        
        # Google AI-specific metrics
        self.google_metrics = {
            "tokens_used": 0,
            "cost_estimate": 0.0,
            "safety_blocks": 0,
            "content_filters": 0
        }
    
    def get_capabilities(self) -> ProviderCapabilities:
        """Get Google AI provider capabilities."""
        return ProviderCapabilities(
            provider_type=ProviderType.GOOGLE,
            supported_models=[
                "gemini-1.5-pro",
                "gemini-1.5-flash",
                "gemini-1.0-pro",
                "gemini-pro",
                "gemini-pro-vision",
                "text-bison-001",
                "chat-bison-001"
            ],
            model_capabilities={
                "gemini-1.5-pro": [ModelCapability.CHAT_COMPLETION, ModelCapability.STREAMING,
                                  ModelCapability.VISION, ModelCapability.REASONING, 
                                  ModelCapability.CODE_GENERATION],
                "gemini-1.5-flash": [ModelCapability.CHAT_COMPLETION, ModelCapability.STREAMING,
                                    ModelCapability.VISION, ModelCapability.CODE_GENERATION],
                "gemini-1.0-pro": [ModelCapability.CHAT_COMPLETION, ModelCapability.STREAMING,
                                  ModelCapability.CODE_GENERATION],
                "gemini-pro": [ModelCapability.CHAT_COMPLETION, ModelCapability.STREAMING,
                              ModelCapability.CODE_GENERATION],
                "gemini-pro-vision": [ModelCapability.CHAT_COMPLETION, ModelCapability.VISION],
                "text-bison-001": [ModelCapability.TEXT_GENERATION],
                "chat-bison-001": [ModelCapability.CHAT_COMPLETION],
            },
            max_tokens={
                "gemini-1.5-pro": 8192,
                "gemini-1.5-flash": 8192,
                "gemini-1.0-pro": 8192,
                "gemini-pro": 8192,
                "gemini-pro-vision": 8192,
                "text-bison-001": 1024,
                "chat-bison-001": 1024,
            },
            context_windows={
                "gemini-1.5-pro": 2000000,  # 2M tokens
                "gemini-1.5-flash": 1000000,  # 1M tokens
                "gemini-1.0-pro": 32768,
                "gemini-pro": 32768,
                "gemini-pro-vision": 16384,
                "text-bison-001": 8192,
                "chat-bison-001": 8192,
            },
            pricing={
                "gemini-1.5-pro": {"input_per_token": 0.00000125, "output_per_token": 0.000005},
                "gemini-1.5-flash": {"input_per_token": 0.000000075, "output_per_token": 0.0000003},
                "gemini-1.0-pro": {"input_per_token": 0.0000005, "output_per_token": 0.0000015},
                "gemini-pro": {"input_per_token": 0.0000005, "output_per_token": 0.0000015},
            },
            rate_limits={
                "gemini-1.5-pro": {"requests_per_minute": 360, "tokens_per_minute": 32000},
                "gemini-1.5-flash": {"requests_per_minute": 1000, "tokens_per_minute": 1000000},
                "gemini-1.0-pro": {"requests_per_minute": 360, "tokens_per_minute": 32000},
            },
            supports_streaming=True,
            supports_async=True,
            api_version="v1"
        )
    
    async def complete_async(self, request: UnifiedRequest) -> UnifiedResponse:
        """Complete a chat/text generation request asynchronously."""
        try:
            # Convert to Google format
            google_request = request.to_provider_format(ProviderType.GOOGLE)
            
            # Use appropriate model if not specified
            model_name = google_request.get("model", self.default_model)
            
            # Initialize model
            model = genai.GenerativeModel(model_name)
            
            # Extract generation config
            generation_config = google_request.get("generationConfig", {})
            
            # Convert messages to Google format
            contents = google_request.get("contents", [])
            
            # Make the API call
            response = await model.generate_content_async(
                contents=contents,
                generation_config=generation_config
            )
            
            # Convert response to unified format
            unified_response = UnifiedResponse.from_google_format({
                "candidates": [{"content": {"parts": [{"text": response.text}]},
                              "finishReason": self._convert_finish_reason(response.candidates[0].finish_reason)}],
                "model": model_name,
                "usageMetadata": {
                    "promptTokenCount": getattr(response.usage_metadata, 'prompt_token_count', 0),
                    "candidatesTokenCount": getattr(response.usage_metadata, 'candidates_token_count', 0),
                    "totalTokenCount": getattr(response.usage_metadata, 'total_token_count', 0)
                } if hasattr(response, 'usage_metadata') else {}
            })
            
            # Update metrics
            if hasattr(response, 'usage_metadata'):
                self._update_metrics(response.usage_metadata)
            
            return unified_response
                
        except Exception as e:
            logger.error(f"Google AI API request failed: {e}")
            raise
    
    async def complete_stream_async(self, request: UnifiedRequest) -> AsyncIterable[UnifiedResponse]:
        """Complete a streaming chat/text generation request asynchronously."""
        try:
            # Convert to Google format
            google_request = request.to_provider_format(ProviderType.GOOGLE)
            
            model_name = google_request.get("model", self.default_model)
            
            # Initialize model
            model = genai.GenerativeModel(model_name)
            
            # Extract generation config
            generation_config = google_request.get("generationConfig", {})
            contents = google_request.get("contents", [])
            
            # Start streaming
            accumulated_content = ""
            
            response_stream = await model.generate_content_async(
                contents=contents,
                generation_config=generation_config,
                stream=True
            )
            
            async for chunk in response_stream:
                if chunk.text:
                    content = chunk.text
                    accumulated_content += content
                    
                    unified_response = UnifiedResponse(
                        content=content,
                        model=model_name,
                        metadata={"accumulated_content": accumulated_content}
                    )
                    
                    yield unified_response
                        
        except Exception as e:
            logger.error(f"Google AI streaming request failed: {e}")
            raise
    
    def _convert_finish_reason(self, finish_reason) -> Optional[str]:
        """Convert Google finish reason to unified format."""
        if not finish_reason:
            return None
        
        finish_reason_str = str(finish_reason)
        
        # Map Google finish reasons to OpenAI-style reasons
        mapping = {
            "STOP": "stop",
            "MAX_TOKENS": "length", 
            "SAFETY": "content_filter",
            "RECITATION": "content_filter",
            "OTHER": "stop"
        }
        
        return mapping.get(finish_reason_str, "stop")
    
    def _update_metrics(self, usage_metadata):
        """Update Google AI-specific metrics."""
        if usage_metadata:
            prompt_tokens = getattr(usage_metadata, 'prompt_token_count', 0)
            candidates_tokens = getattr(usage_metadata, 'candidates_token_count', 0)
            total_tokens = getattr(usage_metadata, 'total_token_count', 0)
            
            self.google_metrics["tokens_used"] += total_tokens
            
            # Estimate cost
            model = self.default_model
            capabilities = self.get_capabilities()
            pricing = capabilities.pricing.get(model, {"input_per_token": 0, "output_per_token": 0})
            
            estimated_cost = (prompt_tokens * pricing["input_per_token"] + 
                            candidates_tokens * pricing["output_per_token"])
            self.google_metrics["cost_estimate"] += estimated_cost
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get plugin metrics including Google AI-specific metrics."""
        base_metrics = super().get_metrics()
        base_metrics.update({
            "google_metrics": self.google_metrics,
            "provider_specific": {
                "api_version": "v1",
                "client_info": {
                    "timeout": self.timeout,
                    "api_key_configured": bool(self.api_key)
                }
            }
        })
        return base_metrics
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on Google AI API."""
        health_status = super().health_check()
        
        try:
            health_status.update({
                "google_specific": {
                    "api_key_configured": bool(self.api_key),
                    "genai_configured": True
                }
            })
            
            # Try to list models as a health check
            try:
                models = list(genai.list_models())
                health_status["google_specific"]["models_accessible"] = len(models) > 0
            except Exception as e:
                health_status["google_specific"]["models_accessible"] = False
                health_status["google_specific"]["models_error"] = str(e)
            
        except Exception as e:
            health_status["status"] = "error" 
            health_status["error"] = f"Google AI health check failed: {e}"
        
        return health_status
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models from Google AI API."""
        try:
            models = []
            for model in genai.list_models():
                models.append({
                    "name": model.name,
                    "display_name": model.display_name,
                    "description": model.description,
                    "input_token_limit": model.input_token_limit,
                    "output_token_limit": model.output_token_limit,
                    "supported_generation_methods": model.supported_generation_methods
                })
            return models
        except Exception as e:
            logger.error(f"Failed to list Google AI models: {e}")
            return []
    
    def cleanup(self):
        """Cleanup resources."""
        # Google AI client doesn't require explicit cleanup
        logger.info("Google AI plugin cleanup completed")