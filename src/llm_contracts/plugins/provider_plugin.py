"""Abstract base class for LLM provider plugins.

This module defines the interface that all LLM provider plugins must implement
to provide unified contract enforcement across different providers.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Callable, AsyncIterable, Type
from enum import Enum
import logging

# Import contract framework components
from ..contracts.base import ContractBase, ValidationResult
from ..core.exceptions import ContractViolationError, ValidationError
from ..validators.input_validator import PerformanceOptimizedInputValidator
from ..validators.output_validator import PerformanceOptimizedOutputValidator

logger = logging.getLogger(__name__)


class ProviderType(Enum):
    """Types of LLM providers supported."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    HUGGINGFACE = "huggingface"
    AZURE_OPENAI = "azure_openai"
    COHERE = "cohere"
    CUSTOM = "custom"


class ModelCapability(Enum):
    """Capabilities that a model can support."""
    TEXT_GENERATION = "text_generation"
    CHAT_COMPLETION = "chat_completion"
    STREAMING = "streaming"
    FUNCTION_CALLING = "function_calling"
    VISION = "vision"
    AUDIO = "audio"
    EMBEDDINGS = "embeddings"
    FINE_TUNING = "fine_tuning"
    CODE_GENERATION = "code_generation"
    REASONING = "reasoning"


@dataclass
class ProviderCapabilities:
    """Capabilities and metadata for a provider."""
    provider_type: ProviderType
    supported_models: List[str] = field(default_factory=list)
    model_capabilities: Dict[str, List[ModelCapability]] = field(default_factory=dict)
    max_tokens: Dict[str, int] = field(default_factory=dict)
    context_windows: Dict[str, int] = field(default_factory=dict)
    pricing: Dict[str, Dict[str, float]] = field(default_factory=dict)  # model -> {input_per_token, output_per_token}
    rate_limits: Dict[str, Dict[str, int]] = field(default_factory=dict)  # model -> {requests_per_minute, tokens_per_minute}
    supports_streaming: bool = True
    supports_async: bool = True
    api_version: str = "v1"
    
    def get_model_max_tokens(self, model: str) -> int:
        """Get maximum tokens for a model."""
        return self.max_tokens.get(model, 4096)
    
    def get_model_context_window(self, model: str) -> int:
        """Get context window size for a model."""
        return self.context_windows.get(model, 4096)
    
    def supports_capability(self, model: str, capability: ModelCapability) -> bool:
        """Check if a model supports a specific capability."""
        return capability in self.model_capabilities.get(model, [])


@dataclass
class UnifiedRequest:
    """Unified request format across all providers."""
    messages: List[Dict[str, Any]]
    model: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False
    functions: Optional[List[Dict[str, Any]]] = None
    function_call: Optional[Union[str, Dict[str, Any]]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    response_format: Optional[Dict[str, Any]] = None
    seed: Optional[int] = None
    user: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {}
        for key, value in self.__dict__.items():
            if value is not None:
                result[key] = value
        return result
    
    @classmethod
    def from_openai_format(cls, request: Dict[str, Any]) -> 'UnifiedRequest':
        """Create from OpenAI API format."""
        return cls(**request)
    
    def to_provider_format(self, provider_type: ProviderType) -> Dict[str, Any]:
        """Convert to provider-specific format."""
        if provider_type == ProviderType.OPENAI:
            return self.to_dict()
        elif provider_type == ProviderType.ANTHROPIC:
            return self._to_anthropic_format()
        elif provider_type == ProviderType.GOOGLE:
            return self._to_google_format()
        else:
            return self.to_dict()
    
    def _to_anthropic_format(self) -> Dict[str, Any]:
        """Convert to Anthropic Claude format."""
        # Convert messages to Anthropic format
        anthropic_messages = []
        system_message = None
        
        for msg in self.messages:
            if msg.get("role") == "system":
                system_message = msg.get("content", "")
            else:
                anthropic_messages.append(msg)
        
        result = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": self.max_tokens or 1000,
        }
        
        if system_message:
            result["system"] = system_message
        
        if self.temperature is not None:
            result["temperature"] = self.temperature
        if self.top_p is not None:
            result["top_p"] = self.top_p
        if self.stop:
            result["stop_sequences"] = self.stop if isinstance(self.stop, list) else [self.stop]
        if self.stream:
            result["stream"] = self.stream
        
        return result
    
    def _to_google_format(self) -> Dict[str, Any]:
        """Convert to Google PaLM/Gemini format."""
        # Convert messages to Google format
        google_contents = []
        for msg in self.messages:
            role = "user" if msg.get("role") == "user" else "model"
            google_contents.append({
                "role": role,
                "parts": [{"text": msg.get("content", "")}]
            })
        
        result = {
            "model": self.model,
            "contents": google_contents,
        }
        
        if self.max_tokens:
            result["generationConfig"] = {"maxOutputTokens": self.max_tokens}
        if self.temperature is not None:
            if "generationConfig" not in result:
                result["generationConfig"] = {}
            result["generationConfig"]["temperature"] = self.temperature
        
        return result


@dataclass
class UnifiedResponse:
    """Unified response format across all providers."""
    content: str
    model: str
    usage: Dict[str, int] = field(default_factory=dict)
    finish_reason: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    response_id: Optional[str] = None
    created: Optional[int] = None
    system_fingerprint: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI response format."""
        message = {
            "role": "assistant",
            "content": self.content
        }
        
        if self.function_call:
            message["function_call"] = self.function_call
        if self.tool_calls:
            message["tool_calls"] = self.tool_calls
        
        return {
            "id": self.response_id or f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": self.created or int(time.time()),
            "model": self.model,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": self.finish_reason or "stop"
            }],
            "usage": self.usage,
            "system_fingerprint": self.system_fingerprint
        }
    
    @classmethod
    def from_openai_format(cls, response: Dict[str, Any]) -> 'UnifiedResponse':
        """Create from OpenAI response format."""
        choice = response.get("choices", [{}])[0]
        message = choice.get("message", {})
        
        return cls(
            content=message.get("content", ""),
            model=response.get("model", ""),
            usage=response.get("usage", {}),
            finish_reason=choice.get("finish_reason"),
            function_call=message.get("function_call"),
            tool_calls=message.get("tool_calls"),
            response_id=response.get("id"),
            created=response.get("created"),
            system_fingerprint=response.get("system_fingerprint")
        )
    
    @classmethod
    def from_anthropic_format(cls, response: Dict[str, Any]) -> 'UnifiedResponse':
        """Create from Anthropic response format."""
        content = ""
        if "content" in response and response["content"]:
            content = response["content"][0].get("text", "")
        
        usage = {}
        if "usage" in response:
            usage = {
                "prompt_tokens": response["usage"].get("input_tokens", 0),
                "completion_tokens": response["usage"].get("output_tokens", 0),
                "total_tokens": response["usage"].get("input_tokens", 0) + response["usage"].get("output_tokens", 0)
            }
        
        return cls(
            content=content,
            model=response.get("model", ""),
            usage=usage,
            finish_reason=response.get("stop_reason"),
            response_id=response.get("id")
        )
    
    @classmethod
    def from_google_format(cls, response: Dict[str, Any]) -> 'UnifiedResponse':
        """Create from Google response format."""
        content = ""
        if "candidates" in response and response["candidates"]:
            candidate = response["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                content = candidate["content"]["parts"][0].get("text", "")
        
        usage = {}
        if "usageMetadata" in response:
            usage_meta = response["usageMetadata"]
            usage = {
                "prompt_tokens": usage_meta.get("promptTokenCount", 0),
                "completion_tokens": usage_meta.get("candidatesTokenCount", 0),
                "total_tokens": usage_meta.get("totalTokenCount", 0)
            }
        
        return cls(
            content=content,
            model=response.get("model", ""),
            usage=usage,
            finish_reason=response.get("candidates", [{}])[0].get("finishReason")
        )


class ProviderPlugin(ABC):
    """Abstract base class for LLM provider plugins."""
    
    def __init__(self,
                 provider_type: ProviderType,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 default_model: Optional[str] = None,
                 contract_enabled: bool = True):
        """Initialize provider plugin.
        
        Args:
            provider_type: Type of the provider
            api_key: API key for authentication
            base_url: Base URL for the API
            default_model: Default model to use
            contract_enabled: Whether contract enforcement is enabled
        """
        self.provider_type = provider_type
        self.api_key = api_key
        self.base_url = base_url
        self.default_model = default_model
        self.contract_enabled = contract_enabled
        
        # Plugin metrics
        self.plugin_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "contract_violations": 0,
            "auto_fixes_applied": 0,
            "average_response_time": 0.0,
            "response_times": []
        }
        
        # Contract validators
        self.input_validator = None
        self.output_validator = None
        
        if self.contract_enabled:
            self._initialize_validators()
    
    def _initialize_validators(self):
        """Initialize contract validators."""
        self.input_validator = PerformanceOptimizedInputValidator()
        self.output_validator = PerformanceOptimizedOutputValidator()
    
    @abstractmethod
    def get_capabilities(self) -> ProviderCapabilities:
        """Get provider capabilities and supported models."""
        pass
    
    @abstractmethod
    async def complete_async(self, request: UnifiedRequest) -> UnifiedResponse:
        """Complete a chat/text generation request asynchronously."""
        pass
    
    @abstractmethod
    async def complete_stream_async(self, request: UnifiedRequest) -> AsyncIterable[UnifiedResponse]:
        """Complete a streaming chat/text generation request asynchronously."""
        pass
    
    def complete(self, request: UnifiedRequest) -> UnifiedResponse:
        """Complete a chat/text generation request synchronously."""
        return asyncio.run(self.complete_async(request))
    
    def complete_stream(self, request: UnifiedRequest) -> AsyncIterable[UnifiedResponse]:
        """Complete a streaming chat/text generation request synchronously."""
        return asyncio.run(self.complete_stream_async(request))
    
    async def complete_with_contracts(self, request: UnifiedRequest) -> UnifiedResponse:
        """Complete request with contract enforcement."""
        start_time = time.time()
        self.plugin_metrics["total_requests"] += 1
        
        try:
            # Input validation
            if self.contract_enabled and self.input_validator:
                input_result = await self.input_validator.validate_async(request.to_dict())
                if not input_result.is_valid:
                    self.plugin_metrics["contract_violations"] += 1
                    if input_result.auto_fixed_content:
                        self.plugin_metrics["auto_fixes_applied"] += 1
                        # Use auto-fixed request
                        request = UnifiedRequest.from_openai_format(input_result.auto_fixed_content)
                    else:
                        raise ContractViolationError(f"Input validation failed: {input_result.error_message}")
            
            # Execute request
            response = await self.complete_async(request)
            
            # Output validation
            if self.contract_enabled and self.output_validator:
                output_result = await self.output_validator.validate_async(response.to_openai_format())
                if not output_result.is_valid:
                    self.plugin_metrics["contract_violations"] += 1
                    if output_result.auto_fixed_content:
                        self.plugin_metrics["auto_fixes_applied"] += 1
                        # Use auto-fixed response
                        response = UnifiedResponse.from_openai_format(output_result.auto_fixed_content)
                    else:
                        raise ContractViolationError(f"Output validation failed: {output_result.error_message}")
            
            # Record success metrics
            response_time = time.time() - start_time
            self.plugin_metrics["successful_requests"] += 1
            self.plugin_metrics["response_times"].append(response_time)
            self._update_average_response_time()
            
            return response
            
        except Exception as e:
            self.plugin_metrics["failed_requests"] += 1
            logger.error(f"Request failed for {self.provider_type.value}: {e}")
            raise
    
    async def complete_stream_with_contracts(self, request: UnifiedRequest) -> AsyncIterable[UnifiedResponse]:
        """Complete streaming request with contract enforcement."""
        start_time = time.time()
        self.plugin_metrics["total_requests"] += 1
        
        try:
            # Input validation
            if self.contract_enabled and self.input_validator:
                input_result = await self.input_validator.validate_async(request.to_dict())
                if not input_result.is_valid:
                    self.plugin_metrics["contract_violations"] += 1
                    if input_result.auto_fixed_content:
                        self.plugin_metrics["auto_fixes_applied"] += 1
                        request = UnifiedRequest.from_openai_format(input_result.auto_fixed_content)
                    else:
                        raise ContractViolationError(f"Input validation failed: {input_result.error_message}")
            
            # Execute streaming request
            full_content = ""
            async for chunk in self.complete_stream_async(request):
                full_content += chunk.content
                
                # Incremental output validation for streaming
                if self.contract_enabled and self.output_validator:
                    # Create a temporary response for validation
                    temp_response = UnifiedResponse(
                        content=full_content,
                        model=chunk.model,
                        usage=chunk.usage
                    )
                    
                    output_result = await self.output_validator.validate_async(temp_response.to_openai_format())
                    if not output_result.is_valid and output_result.error_message:
                        # Check for critical violations that should stop streaming
                        if "critical" in output_result.error_message.lower():
                            self.plugin_metrics["contract_violations"] += 1
                            raise ContractViolationError(f"Critical violation detected: {output_result.error_message}")
                
                yield chunk
            
            # Final validation
            if self.contract_enabled and self.output_validator:
                final_response = UnifiedResponse(
                    content=full_content,
                    model=request.model,
                    usage={}
                )
                output_result = await self.output_validator.validate_async(final_response.to_openai_format())
                if not output_result.is_valid:
                    self.plugin_metrics["contract_violations"] += 1
                    logger.warning(f"Final output validation failed: {output_result.error_message}")
            
            # Record success metrics
            response_time = time.time() - start_time
            self.plugin_metrics["successful_requests"] += 1
            self.plugin_metrics["response_times"].append(response_time)
            self._update_average_response_time()
            
        except Exception as e:
            self.plugin_metrics["failed_requests"] += 1
            logger.error(f"Streaming request failed for {self.provider_type.value}: {e}")
            raise
    
    def add_contract(self, contract: ContractBase, contract_type: str = "output"):
        """Add a contract to the plugin."""
        if contract_type == "input" and self.input_validator:
            self.input_validator.add_contract(contract)
        elif contract_type == "output" and self.output_validator:
            self.output_validator.add_contract(contract)
    
    def remove_contract(self, contract_name: str, contract_type: str = "output"):
        """Remove a contract from the plugin."""
        if contract_type == "input" and self.input_validator:
            self.input_validator.remove_contract(contract_name)
        elif contract_type == "output" and self.output_validator:
            self.output_validator.remove_contract(contract_name)
    
    def _update_average_response_time(self):
        """Update average response time metric."""
        if self.plugin_metrics["response_times"]:
            self.plugin_metrics["average_response_time"] = sum(self.plugin_metrics["response_times"]) / len(self.plugin_metrics["response_times"])
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get plugin metrics."""
        metrics = self.plugin_metrics.copy()
        
        # Calculate additional metrics
        total_requests = metrics["total_requests"]
        if total_requests > 0:
            metrics["success_rate"] = metrics["successful_requests"] / total_requests
            metrics["failure_rate"] = metrics["failed_requests"] / total_requests
            metrics["violation_rate"] = metrics["contract_violations"] / total_requests
        else:
            metrics["success_rate"] = 0.0
            metrics["failure_rate"] = 0.0
            metrics["violation_rate"] = 0.0
        
        return metrics
    
    def reset_metrics(self):
        """Reset plugin metrics."""
        self.plugin_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "contract_violations": 0,
            "auto_fixes_applied": 0,
            "average_response_time": 0.0,
            "response_times": []
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the provider."""
        return {
            "provider_type": self.provider_type.value,
            "status": "healthy",
            "api_accessible": True,
            "contract_enabled": self.contract_enabled,
            "metrics": self.get_metrics()
        }
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.provider_type.value})"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(provider_type={self.provider_type}, model={self.default_model})"