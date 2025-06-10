"""Plugin manager for LLM provider plugins.

This module provides centralized management of LLM provider plugins,
including registration, discovery, lifecycle management, and routing.
"""

import asyncio
import importlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, Union, Callable
from enum import Enum
import logging
import threading
from pathlib import Path

from .provider_plugin import ProviderPlugin, ProviderType, UnifiedRequest, UnifiedResponse

logger = logging.getLogger(__name__)


class PluginStatus(Enum):
    """Plugin status states."""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    ACTIVE = "active"
    ERROR = "error"
    DISABLED = "disabled"


@dataclass
class PluginConfig:
    """Configuration for a plugin."""
    name: str
    provider_type: ProviderType
    plugin_class: str
    module_path: str
    enabled: bool = True
    priority: int = 100  # Lower number = higher priority
    config: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    auto_load: bool = True
    health_check_interval: int = 300  # seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "provider_type": self.provider_type.value,
            "plugin_class": self.plugin_class,
            "module_path": self.module_path,
            "enabled": self.enabled,
            "priority": self.priority,
            "config": self.config,
            "dependencies": self.dependencies,
            "auto_load": self.auto_load,
            "health_check_interval": self.health_check_interval
        }


@dataclass
class PluginInfo:
    """Information about a loaded plugin."""
    config: PluginConfig
    instance: Optional[ProviderPlugin] = None
    status: PluginStatus = PluginStatus.UNLOADED
    last_error: Optional[str] = None
    load_time: Optional[float] = None
    last_health_check: Optional[float] = None
    health_status: Dict[str, Any] = field(default_factory=dict)
    
    def is_healthy(self) -> bool:
        """Check if plugin is healthy."""
        return (self.status == PluginStatus.ACTIVE and 
                self.instance is not None and
                self.health_status.get("status") == "healthy")


class PluginManager:
    """Centralized manager for LLM provider plugins."""
    
    def __init__(self, auto_discover: bool = True):
        """Initialize plugin manager.
        
        Args:
            auto_discover: Whether to automatically discover plugins
        """
        self.plugins: Dict[str, PluginInfo] = {}
        self.provider_map: Dict[ProviderType, str] = {}  # provider_type -> plugin_name
        self.auto_discover = auto_discover
        
        # Plugin lifecycle hooks
        self.hooks = {
            "pre_load": [],
            "post_load": [],
            "pre_unload": [],
            "post_unload": [],
            "health_check": []
        }
        
        # Health check management
        self._health_check_task = None
        self._health_check_running = False
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Manager metrics
        self.manager_metrics = {
            "plugins_loaded": 0,
            "plugins_failed": 0,
            "total_requests": 0,
            "requests_by_provider": {},
            "health_checks_performed": 0,
            "auto_discoveries": 0
        }
        
        if self.auto_discover:
            self.discover_plugins()
    
    def register_plugin(self, config: PluginConfig) -> bool:
        """Register a plugin configuration.
        
        Args:
            config: Plugin configuration
            
        Returns:
            True if registration successful, False otherwise
        """
        with self._lock:
            try:
                if config.name in self.plugins:
                    logger.warning(f"Plugin {config.name} already registered, updating configuration")
                
                plugin_info = PluginInfo(config=config)
                self.plugins[config.name] = plugin_info
                
                # Update provider mapping
                self.provider_map[config.provider_type] = config.name
                
                logger.info(f"Registered plugin: {config.name} ({config.provider_type.value})")
                
                # Auto-load if enabled
                if config.auto_load and config.enabled:
                    return self.load_plugin(config.name)
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to register plugin {config.name}: {e}")
                return False
    
    def unregister_plugin(self, plugin_name: str) -> bool:
        """Unregister a plugin.
        
        Args:
            plugin_name: Name of the plugin to unregister
            
        Returns:
            True if unregistration successful, False otherwise
        """
        with self._lock:
            try:
                if plugin_name not in self.plugins:
                    logger.warning(f"Plugin {plugin_name} not registered")
                    return False
                
                plugin_info = self.plugins[plugin_name]
                
                # Unload plugin if loaded
                if plugin_info.status in [PluginStatus.LOADED, PluginStatus.ACTIVE]:
                    self.unload_plugin(plugin_name)
                
                # Remove from mappings
                provider_type = plugin_info.config.provider_type
                if provider_type in self.provider_map and self.provider_map[provider_type] == plugin_name:
                    del self.provider_map[provider_type]
                
                del self.plugins[plugin_name]
                
                logger.info(f"Unregistered plugin: {plugin_name}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to unregister plugin {plugin_name}: {e}")
                return False
    
    def load_plugin(self, plugin_name: str) -> bool:
        """Load a plugin.
        
        Args:
            plugin_name: Name of the plugin to load
            
        Returns:
            True if loading successful, False otherwise
        """
        with self._lock:
            try:
                if plugin_name not in self.plugins:
                    logger.error(f"Plugin {plugin_name} not registered")
                    return False
                
                plugin_info = self.plugins[plugin_name]
                
                if plugin_info.status == PluginStatus.ACTIVE:
                    logger.info(f"Plugin {plugin_name} already loaded")
                    return True
                
                if not plugin_info.config.enabled:
                    logger.warning(f"Plugin {plugin_name} is disabled")
                    return False
                
                # Execute pre-load hooks
                self._execute_hooks("pre_load", plugin_info)
                
                # Update status
                plugin_info.status = PluginStatus.LOADING
                start_time = time.time()
                
                # Load the plugin module and class
                module = importlib.import_module(plugin_info.config.module_path)
                plugin_class = getattr(module, plugin_info.config.plugin_class)
                
                # Instantiate the plugin
                plugin_instance = plugin_class(**plugin_info.config.config)
                
                if not isinstance(plugin_instance, ProviderPlugin):
                    raise TypeError(f"Plugin {plugin_name} must inherit from ProviderPlugin")
                
                plugin_info.instance = plugin_instance
                plugin_info.status = PluginStatus.ACTIVE
                plugin_info.load_time = time.time() - start_time
                plugin_info.last_error = None
                
                # Execute post-load hooks
                self._execute_hooks("post_load", plugin_info)
                
                self.manager_metrics["plugins_loaded"] += 1
                logger.info(f"Loaded plugin: {plugin_name} in {plugin_info.load_time:.3f}s")
                
                return True
                
            except Exception as e:
                plugin_info.status = PluginStatus.ERROR
                plugin_info.last_error = str(e)
                self.manager_metrics["plugins_failed"] += 1
                logger.error(f"Failed to load plugin {plugin_name}: {e}")
                return False
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a plugin.
        
        Args:
            plugin_name: Name of the plugin to unload
            
        Returns:
            True if unloading successful, False otherwise
        """
        with self._lock:
            try:
                if plugin_name not in self.plugins:
                    logger.error(f"Plugin {plugin_name} not registered")
                    return False
                
                plugin_info = self.plugins[plugin_name]
                
                if plugin_info.status == PluginStatus.UNLOADED:
                    logger.info(f"Plugin {plugin_name} already unloaded")
                    return True
                
                # Execute pre-unload hooks
                self._execute_hooks("pre_unload", plugin_info)
                
                # Cleanup plugin instance
                if plugin_info.instance:
                    # Call cleanup method if available
                    if hasattr(plugin_info.instance, 'cleanup'):
                        plugin_info.instance.cleanup()
                    
                    plugin_info.instance = None
                
                plugin_info.status = PluginStatus.UNLOADED
                plugin_info.last_error = None
                
                # Execute post-unload hooks
                self._execute_hooks("post_unload", plugin_info)
                
                logger.info(f"Unloaded plugin: {plugin_name}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to unload plugin {plugin_name}: {e}")
                return False
    
    def get_plugin(self, plugin_name: str) -> Optional[ProviderPlugin]:
        """Get a loaded plugin instance.
        
        Args:
            plugin_name: Name of the plugin
            
        Returns:
            Plugin instance or None if not found/loaded
        """
        with self._lock:
            plugin_info = self.plugins.get(plugin_name)
            if plugin_info and plugin_info.status == PluginStatus.ACTIVE:
                return plugin_info.instance
            return None
    
    def get_provider(self, provider_type: ProviderType) -> Optional[ProviderPlugin]:
        """Get a plugin by provider type.
        
        Args:
            provider_type: Type of provider
            
        Returns:
            Plugin instance or None if not found
        """
        plugin_name = self.provider_map.get(provider_type)
        if plugin_name:
            return self.get_plugin(plugin_name)
        return None
    
    def list_plugins(self, status_filter: Optional[PluginStatus] = None) -> List[PluginInfo]:
        """List all plugins, optionally filtered by status.
        
        Args:
            status_filter: Optional status to filter by
            
        Returns:
            List of plugin information
        """
        with self._lock:
            plugins = list(self.plugins.values())
            if status_filter:
                plugins = [p for p in plugins if p.status == status_filter]
            return sorted(plugins, key=lambda p: p.config.priority)
    
    def list_providers(self, active_only: bool = True) -> List[ProviderType]:
        """List available provider types.
        
        Args:
            active_only: Whether to only include active providers
            
        Returns:
            List of provider types
        """
        with self._lock:
            if active_only:
                return [pt for pt, name in self.provider_map.items() 
                       if self.plugins.get(name, {}).status == PluginStatus.ACTIVE]
            else:
                return list(self.provider_map.keys())
    
    def discover_plugins(self, search_paths: Optional[List[str]] = None) -> int:
        """Discover plugins automatically.
        
        Args:
            search_paths: Optional paths to search for plugins
            
        Returns:
            Number of plugins discovered
        """
        discovered_count = 0
        
        if search_paths is None:
            # Default search paths
            search_paths = [
                "llm_contracts.plugins",
                "llm_contracts.plugins.providers"
            ]
        
        for search_path in search_paths:
            try:
                discovered_count += self._discover_in_module(search_path)
            except Exception as e:
                logger.debug(f"Failed to discover plugins in {search_path}: {e}")
        
        self.manager_metrics["auto_discoveries"] += 1
        logger.info(f"Discovered {discovered_count} plugins")
        
        return discovered_count
    
    def _discover_in_module(self, module_path: str) -> int:
        """Discover plugins in a specific module."""
        discovered = 0
        
        try:
            module = importlib.import_module(module_path)
            
            # Look for plugin classes
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                
                if (isinstance(attr, type) and 
                    issubclass(attr, ProviderPlugin) and 
                    attr != ProviderPlugin):
                    
                    # Try to auto-register
                    plugin_name = attr.__name__.lower().replace("plugin", "")
                    
                    # Try to determine provider type from class name
                    provider_type = self._infer_provider_type(attr.__name__)
                    
                    if provider_type and plugin_name not in self.plugins:
                        config = PluginConfig(
                            name=plugin_name,
                            provider_type=provider_type,
                            plugin_class=attr.__name__,
                            module_path=module_path,
                            auto_load=False  # Don't auto-load discovered plugins
                        )
                        
                        self.register_plugin(config)
                        discovered += 1
            
        except ImportError:
            # Module doesn't exist or can't be imported
            pass
        except Exception as e:
            logger.debug(f"Error discovering plugins in {module_path}: {e}")
        
        return discovered
    
    def _infer_provider_type(self, class_name: str) -> Optional[ProviderType]:
        """Infer provider type from class name."""
        name_lower = class_name.lower()
        
        if "openai" in name_lower:
            return ProviderType.OPENAI
        elif "anthropic" in name_lower or "claude" in name_lower:
            return ProviderType.ANTHROPIC
        elif "google" in name_lower or "palm" in name_lower or "gemini" in name_lower:
            return ProviderType.GOOGLE
        elif "huggingface" in name_lower or "hf" in name_lower:
            return ProviderType.HUGGINGFACE
        elif "azure" in name_lower:
            return ProviderType.AZURE_OPENAI
        elif "cohere" in name_lower:
            return ProviderType.COHERE
        
        return None
    
    async def route_request(self, request: UnifiedRequest, provider_type: Optional[ProviderType] = None) -> UnifiedResponse:
        """Route a request to the appropriate provider.
        
        Args:
            request: Unified request
            provider_type: Optional specific provider to use
            
        Returns:
            Unified response
        """
        # Determine provider
        if provider_type is None:
            # Use default routing logic (e.g., based on model name)
            provider_type = self._infer_provider_from_model(request.model)
        
        provider = self.get_provider(provider_type)
        if not provider:
            raise ValueError(f"No active provider found for {provider_type}")
        
        # Update metrics
        self.manager_metrics["total_requests"] += 1
        provider_name = provider_type.value
        if provider_name not in self.manager_metrics["requests_by_provider"]:
            self.manager_metrics["requests_by_provider"][provider_name] = 0
        self.manager_metrics["requests_by_provider"][provider_name] += 1
        
        # Route to provider
        return await provider.complete_with_contracts(request)
    
    async def route_stream_request(self, request: UnifiedRequest, provider_type: Optional[ProviderType] = None):
        """Route a streaming request to the appropriate provider.
        
        Args:
            request: Unified request
            provider_type: Optional specific provider to use
            
        Returns:
            Async generator of unified responses
        """
        # Determine provider
        if provider_type is None:
            provider_type = self._infer_provider_from_model(request.model)
        
        provider = self.get_provider(provider_type)
        if not provider:
            raise ValueError(f"No active provider found for {provider_type}")
        
        # Update metrics
        self.manager_metrics["total_requests"] += 1
        provider_name = provider_type.value
        if provider_name not in self.manager_metrics["requests_by_provider"]:
            self.manager_metrics["requests_by_provider"][provider_name] = 0
        self.manager_metrics["requests_by_provider"][provider_name] += 1
        
        # Route to provider
        async for response in provider.complete_stream_with_contracts(request):
            yield response
    
    def _infer_provider_from_model(self, model: str) -> ProviderType:
        """Infer provider type from model name."""
        model_lower = model.lower()
        
        if any(x in model_lower for x in ["gpt", "davinci", "curie", "babbage", "ada"]):
            return ProviderType.OPENAI
        elif any(x in model_lower for x in ["claude", "anthropic"]):
            return ProviderType.ANTHROPIC
        elif any(x in model_lower for x in ["palm", "gemini", "bison"]):
            return ProviderType.GOOGLE
        elif any(x in model_lower for x in ["llama", "mistral", "falcon"]):
            return ProviderType.HUGGINGFACE
        else:
            # Default to OpenAI
            return ProviderType.OPENAI
    
    def add_hook(self, hook_type: str, callback: Callable):
        """Add a lifecycle hook.
        
        Args:
            hook_type: Type of hook (pre_load, post_load, etc.)
            callback: Callback function
        """
        if hook_type in self.hooks:
            self.hooks[hook_type].append(callback)
    
    def remove_hook(self, hook_type: str, callback: Callable):
        """Remove a lifecycle hook.
        
        Args:
            hook_type: Type of hook
            callback: Callback function to remove
        """
        if hook_type in self.hooks and callback in self.hooks[hook_type]:
            self.hooks[hook_type].remove(callback)
    
    def _execute_hooks(self, hook_type: str, plugin_info: PluginInfo):
        """Execute lifecycle hooks."""
        for callback in self.hooks.get(hook_type, []):
            try:
                callback(plugin_info)
            except Exception as e:
                logger.error(f"Hook {hook_type} failed for plugin {plugin_info.config.name}: {e}")
    
    async def start_health_monitoring(self, interval: int = 300):
        """Start health monitoring for all plugins.
        
        Args:
            interval: Health check interval in seconds
        """
        if self._health_check_running:
            logger.warning("Health monitoring already running")
            return
        
        self._health_check_running = True
        self._health_check_task = asyncio.create_task(self._health_check_loop(interval))
        logger.info(f"Started health monitoring with {interval}s interval")
    
    async def stop_health_monitoring(self):
        """Stop health monitoring."""
        if self._health_check_task:
            self._health_check_running = False
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_check_task = None
            logger.info("Stopped health monitoring")
    
    async def _health_check_loop(self, interval: int):
        """Health check loop."""
        while self._health_check_running:
            try:
                await self.perform_health_checks()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(interval)
    
    async def perform_health_checks(self):
        """Perform health checks on all active plugins."""
        with self._lock:
            active_plugins = [info for info in self.plugins.values() 
                            if info.status == PluginStatus.ACTIVE and info.instance]
        
        for plugin_info in active_plugins:
            try:
                health_status = plugin_info.instance.health_check()
                plugin_info.health_status = health_status
                plugin_info.last_health_check = time.time()
                
                # Execute health check hooks
                self._execute_hooks("health_check", plugin_info)
                
                # Log unhealthy plugins
                if health_status.get("status") != "healthy":
                    logger.warning(f"Plugin {plugin_info.config.name} unhealthy: {health_status}")
                
                self.manager_metrics["health_checks_performed"] += 1
                
            except Exception as e:
                logger.error(f"Health check failed for {plugin_info.config.name}: {e}")
                plugin_info.health_status = {"status": "error", "error": str(e)}
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get manager metrics."""
        with self._lock:
            plugin_metrics = {}
            for name, info in self.plugins.items():
                if info.instance:
                    plugin_metrics[name] = info.instance.get_metrics()
            
            return {
                "manager_metrics": self.manager_metrics,
                "plugin_metrics": plugin_metrics,
                "plugin_count": len(self.plugins),
                "active_plugins": len([p for p in self.plugins.values() if p.status == PluginStatus.ACTIVE]),
                "health_check_running": self._health_check_running
            }
    
    async def shutdown(self):
        """Shutdown the plugin manager."""
        logger.info("Shutting down plugin manager")
        
        # Stop health monitoring
        await self.stop_health_monitoring()
        
        # Unload all plugins
        with self._lock:
            plugin_names = list(self.plugins.keys())
        
        for plugin_name in plugin_names:
            self.unload_plugin(plugin_name)
        
        logger.info("Plugin manager shutdown complete")