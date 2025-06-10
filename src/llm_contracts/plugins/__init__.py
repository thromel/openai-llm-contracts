"""Plugin system for multi-provider LLM contract architecture.

This module provides a comprehensive plugin system for integrating multiple
LLM providers (OpenAI, Anthropic, etc.) with unified contract enforcement.
"""

from .plugin_manager import PluginManager, PluginConfig, PluginStatus
from .provider_plugin import ProviderPlugin, ProviderCapabilities, UnifiedRequest, UnifiedResponse
from .openai_plugin import OpenAIPlugin
from .anthropic_plugin import AnthropicPlugin
from .google_plugin import GooglePlugin
from .huggingface_plugin import HuggingFacePlugin
from .azure_plugin import AzureOpenAIPlugin
from .provider_factory import ProviderFactory, create_provider
from .plugin_registry import PluginRegistry

__all__ = [
    # Core plugin system
    'PluginManager',
    'PluginConfig', 
    'PluginStatus',
    
    # Provider plugin base
    'ProviderPlugin',
    'ProviderCapabilities',
    'UnifiedRequest',
    'UnifiedResponse',
    
    # Specific provider plugins
    'OpenAIPlugin',
    'AnthropicPlugin', 
    'GooglePlugin',
    'HuggingFacePlugin',
    'AzureOpenAIPlugin',
    
    # Factory and registry
    'ProviderFactory',
    'PluginRegistry',
    'create_provider'
]