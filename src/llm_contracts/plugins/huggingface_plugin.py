"""HuggingFace provider plugin for the LLM contract framework.

This module provides integration with HuggingFace's Inference API and Transformers
through the plugin system.
"""

import os
import asyncio
import aiohttp
from typing import Any, Dict, List, Optional, AsyncIterable
import logging
import json

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from .provider_plugin import (
    ProviderPlugin, ProviderType, ProviderCapabilities, ModelCapability,
    UnifiedRequest, UnifiedResponse
)

logger = logging.getLogger(__name__)


class HuggingFacePlugin(ProviderPlugin):
    """HuggingFace provider plugin implementation."""
    
    def __init__(self,
                 api_key: Optional[str] = None,
                 base_url: str = "https://api-inference.huggingface.co",
                 default_model: str = "microsoft/DialoGPT-large",
                 contract_enabled: bool = True,
                 timeout: float = 60.0,
                 use_inference_api: bool = True):
        """Initialize HuggingFace plugin.
        
        Args:
            api_key: HuggingFace API token (uses HUGGINGFACE_API_KEY env var if not provided)
            base_url: Base URL for HuggingFace Inference API
            default_model: Default model to use
            contract_enabled: Whether contract enforcement is enabled
            timeout: Request timeout in seconds
            use_inference_api: Whether to use HuggingFace Inference API
        """
        super().__init__(
            provider_type=ProviderType.HUGGINGFACE,
            api_key=api_key or os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN"),
            base_url=base_url,
            default_model=default_model,
            contract_enabled=contract_enabled
        )
        
        self.timeout = timeout
        self.use_inference_api = use_inference_api
        
        # HuggingFace-specific metrics
        self.huggingface_metrics = {
            "tokens_used": 0,
            "model_loads": 0,
            "inference_time": 0.0,
            "api_calls": 0
        }
        
        # Tokenizer cache for token counting
        self.tokenizer_cache = {}
    
    def get_capabilities(self) -> ProviderCapabilities:
        """Get HuggingFace provider capabilities."""
        return ProviderCapabilities(
            provider_type=ProviderType.HUGGINGFACE,
            supported_models=[
                # Popular text generation models
                "microsoft/DialoGPT-large",
                "microsoft/DialoGPT-medium", 
                "microsoft/DialoGPT-small",
                "facebook/blenderbot-400M-distill",
                "facebook/blenderbot-1B-distill",
                "facebook/blenderbot-3B",
                "bigscience/bloom-560m",
                "bigscience/bloom-1b1",
                "bigscience/bloom-3b",
                "bigscience/bloom-7b1",
                "EleutherAI/gpt-neo-125M",
                "EleutherAI/gpt-neo-1.3B",
                "EleutherAI/gpt-neo-2.7B",
                "EleutherAI/gpt-j-6B",
                "meta-llama/Llama-2-7b-chat-hf",
                "meta-llama/Llama-2-13b-chat-hf",
                "mistralai/Mistral-7B-Instruct-v0.1",
                "mistralai/Mixtral-8x7B-Instruct-v0.1",
                "tiiuae/falcon-7b-instruct",
                "tiiuae/falcon-40b-instruct",
                "HuggingFaceH4/zephyr-7b-beta",
                "openchat/openchat-3.5-1210",
                "teknium/OpenHermes-2.5-Mistral-7B"
            ],
            model_capabilities={
                "microsoft/DialoGPT-large": [ModelCapability.CHAT_COMPLETION, ModelCapability.TEXT_GENERATION],
                "facebook/blenderbot-3B": [ModelCapability.CHAT_COMPLETION],
                "bigscience/bloom-7b1": [ModelCapability.TEXT_GENERATION, ModelCapability.CODE_GENERATION],
                "EleutherAI/gpt-j-6B": [ModelCapability.TEXT_GENERATION, ModelCapability.CODE_GENERATION],
                "meta-llama/Llama-2-7b-chat-hf": [ModelCapability.CHAT_COMPLETION, ModelCapability.REASONING],
                "mistralai/Mistral-7B-Instruct-v0.1": [ModelCapability.CHAT_COMPLETION, ModelCapability.CODE_GENERATION],
                "tiiuae/falcon-7b-instruct": [ModelCapability.CHAT_COMPLETION, ModelCapability.CODE_GENERATION],
            },
            max_tokens={
                "microsoft/DialoGPT-large": 1024,
                "facebook/blenderbot-3B": 512,
                "bigscience/bloom-7b1": 2048,
                "EleutherAI/gpt-j-6B": 2048,
                "meta-llama/Llama-2-7b-chat-hf": 4096,
                "mistralai/Mistral-7B-Instruct-v0.1": 8192,
                "tiiuae/falcon-7b-instruct": 2048,
            },
            context_windows={
                "microsoft/DialoGPT-large": 1024,
                "facebook/blenderbot-3B": 512,
                "bigscience/bloom-7b1": 2048,
                "EleutherAI/gpt-j-6B": 2048,
                "meta-llama/Llama-2-7b-chat-hf": 4096,
                "mistralai/Mistral-7B-Instruct-v0.1": 8192,
                "tiiuae/falcon-7b-instruct": 2048,
            },
            pricing={
                # HuggingFace Inference API is generally free for small usage
                # These are placeholder values
                "microsoft/DialoGPT-large": {"input_per_token": 0.0, "output_per_token": 0.0},
                "meta-llama/Llama-2-7b-chat-hf": {"input_per_token": 0.0, "output_per_token": 0.0},
            },
            rate_limits={
                # Rate limits vary by model and account type
                "microsoft/DialoGPT-large": {"requests_per_minute": 30, "tokens_per_minute": 30000},
                "meta-llama/Llama-2-7b-chat-hf": {"requests_per_minute": 10, "tokens_per_minute": 10000},
            },
            supports_streaming=False,  # HuggingFace Inference API doesn't support streaming
            supports_async=True,
            api_version="v1"
        )
    
    async def complete_async(self, request: UnifiedRequest) -> UnifiedResponse:
        """Complete a chat/text generation request asynchronously."""
        try:
            model = request.model or self.default_model
            
            if self.use_inference_api:
                return await self._complete_via_inference_api(request, model)
            else:
                # Local transformers usage would go here
                raise NotImplementedError("Local transformers support not implemented yet")
                
        except Exception as e:
            logger.error(f"HuggingFace API request failed: {e}")
            raise
    
    async def _complete_via_inference_api(self, request: UnifiedRequest, model: str) -> UnifiedResponse:
        """Complete request using HuggingFace Inference API."""
        import time
        start_time = time.time()
        
        # Prepare the request
        if request.messages:
            # Convert messages to a single text prompt
            text = self._messages_to_text(request.messages)
        else:
            text = str(request.messages[0].get('content', '')) if request.messages else ""
        
        # Prepare payload
        payload = {
            "inputs": text,
            "parameters": {
                "max_new_tokens": request.max_tokens or 100,
                "temperature": request.temperature or 0.7,
                "top_p": request.top_p or 0.9,
                "do_sample": True,
                "return_full_text": False
            }
        }
        
        # Make API request
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers["Content-Type"] = "application/json"
        
        url = f"{self.base_url}/models/{model}"
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    # Extract generated text
                    if isinstance(result, list) and len(result) > 0:
                        generated_text = result[0].get("generated_text", "")
                    else:
                        generated_text = str(result)
                    
                    # Estimate token usage
                    input_tokens = self._estimate_tokens(text, model)
                    output_tokens = self._estimate_tokens(generated_text, model)
                    
                    # Update metrics
                    inference_time = time.time() - start_time
                    self.huggingface_metrics["api_calls"] += 1
                    self.huggingface_metrics["inference_time"] += inference_time
                    self.huggingface_metrics["tokens_used"] += input_tokens + output_tokens
                    
                    return UnifiedResponse(
                        content=generated_text,
                        model=model,
                        usage={
                            "prompt_tokens": input_tokens,
                            "completion_tokens": output_tokens,
                            "total_tokens": input_tokens + output_tokens
                        },
                        finish_reason="stop",
                        metadata={"inference_time": inference_time}
                    )
                
                else:
                    error_text = await response.text()
                    raise Exception(f"HuggingFace API error {response.status}: {error_text}")
    
    async def complete_stream_async(self, request: UnifiedRequest) -> AsyncIterable[UnifiedResponse]:
        """Complete a streaming request (not supported by HuggingFace Inference API)."""
        # HuggingFace Inference API doesn't support streaming
        # Return the complete response as a single chunk
        response = await self.complete_async(request)
        yield response
    
    def _messages_to_text(self, messages: List[Dict[str, Any]]) -> str:
        """Convert OpenAI-style messages to plain text."""
        text_parts = []
        
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            
            if role == "system":
                text_parts.append(f"System: {content}")
            elif role == "user":
                text_parts.append(f"Human: {content}")
            elif role == "assistant":
                text_parts.append(f"Assistant: {content}")
            else:
                text_parts.append(content)
        
        # Add a prompt for the assistant to respond
        text_parts.append("Assistant:")
        
        return "\n".join(text_parts)
    
    def _estimate_tokens(self, text: str, model: str) -> int:
        """Estimate token count for given text."""
        if TRANSFORMERS_AVAILABLE:
            try:
                # Try to get tokenizer for the model
                if model not in self.tokenizer_cache:
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(model)
                        self.tokenizer_cache[model] = tokenizer
                    except Exception:
                        # Fallback to a default tokenizer
                        try:
                            tokenizer = AutoTokenizer.from_pretrained("gpt2")
                            self.tokenizer_cache[model] = tokenizer
                        except Exception:
                            # Ultimate fallback: rough estimation
                            return len(text.split()) * 1.3
                
                tokenizer = self.tokenizer_cache[model]
                tokens = tokenizer.encode(text)
                return len(tokens)
                
            except Exception:
                pass
        
        # Fallback: rough estimation (1.3 tokens per word on average)
        return int(len(text.split()) * 1.3)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get plugin metrics including HuggingFace-specific metrics."""
        base_metrics = super().get_metrics()
        base_metrics.update({
            "huggingface_metrics": self.huggingface_metrics,
            "provider_specific": {
                "api_version": "v1",
                "use_inference_api": self.use_inference_api,
                "transformers_available": TRANSFORMERS_AVAILABLE,
                "client_info": {
                    "base_url": self.base_url,
                    "timeout": self.timeout
                }
            }
        })
        return base_metrics
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on HuggingFace API."""
        health_status = super().health_check()
        
        try:
            health_status.update({
                "huggingface_specific": {
                    "api_key_configured": bool(self.api_key),
                    "transformers_available": TRANSFORMERS_AVAILABLE,
                    "inference_api_url": self.base_url,
                    "use_inference_api": self.use_inference_api
                }
            })
            
        except Exception as e:
            health_status["status"] = "error"
            health_status["error"] = f"HuggingFace health check failed: {e}"
        
        return health_status
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models (returns configured models)."""
        # HuggingFace has thousands of models, so we return our supported subset
        capabilities = self.get_capabilities()
        return [
            {
                "name": model,
                "capabilities": capabilities.model_capabilities.get(model, []),
                "max_tokens": capabilities.max_tokens.get(model, 1024),
                "context_window": capabilities.context_windows.get(model, 1024)
            }
            for model in capabilities.supported_models
        ]
    
    def cleanup(self):
        """Cleanup resources."""
        # Clear tokenizer cache
        self.tokenizer_cache.clear()
        logger.info("HuggingFace plugin cleanup completed")