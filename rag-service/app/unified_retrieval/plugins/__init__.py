"""Plugin system for the unified retrieval framework."""

from .base import (
    PluginInterface,
    PluginRegistry,
    PluginLoader,
    BasePlugin,
    get_global_registry,
    register_plugin,
    get_strategy
)

__all__ = [
    "PluginInterface",
    "PluginRegistry",
    "PluginLoader",
    "BasePlugin",
    "get_global_registry",
    "register_plugin",
    "get_strategy"
]