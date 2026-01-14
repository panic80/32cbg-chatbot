"""Plugin system for extending the unified retrieval framework."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Type, Callable
import importlib
import inspect
from pathlib import Path

from app.core.logging import get_logger
from app.unified_retrieval.strategies.base import BaseStrategy, StrategyType

logger = get_logger(__name__)


class PluginInterface(ABC):
    """Base interface that all plugins must implement."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name."""
        pass
        
    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version."""
        pass
        
    @property
    @abstractmethod
    def description(self) -> str:
        """Plugin description."""
        pass
        
    @abstractmethod
    def get_strategies(self) -> Dict[str, Type[BaseStrategy]]:
        """
        Get strategy classes provided by this plugin.
        
        Returns:
            Dictionary mapping strategy names to strategy classes
        """
        pass
        
    def get_config_schema(self) -> Optional[Dict[str, Any]]:
        """
        Get configuration schema for this plugin.
        
        Returns:
            JSON schema for plugin configuration
        """
        return None
        
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the plugin with configuration.
        
        Args:
            config: Plugin configuration
        """
        pass
        
    def shutdown(self) -> None:
        """Clean up plugin resources."""
        pass


class PluginRegistry:
    """Registry for managing retrieval plugins."""
    
    def __init__(self):
        """Initialize the plugin registry."""
        self._plugins: Dict[str, PluginInterface] = {}
        self._strategies: Dict[str, Type[BaseStrategy]] = {}
        self._strategy_to_plugin: Dict[str, str] = {}
        
    def register_plugin(self, plugin: PluginInterface) -> None:
        """
        Register a plugin.
        
        Args:
            plugin: Plugin instance to register
        """
        if plugin.name in self._plugins:
            raise ValueError(f"Plugin '{plugin.name}' already registered")
            
        # Register the plugin
        self._plugins[plugin.name] = plugin
        
        # Register its strategies
        strategies = plugin.get_strategies()
        for strategy_name, strategy_class in strategies.items():
            full_name = f"{plugin.name}.{strategy_name}"
            
            if full_name in self._strategies:
                raise ValueError(f"Strategy '{full_name}' already registered")
                
            self._strategies[full_name] = strategy_class
            self._strategy_to_plugin[full_name] = plugin.name
            
        logger.info(
            f"Registered plugin '{plugin.name}' with {len(strategies)} strategies"
        )
        
    def unregister_plugin(self, plugin_name: str) -> None:
        """
        Unregister a plugin.
        
        Args:
            plugin_name: Name of plugin to unregister
        """
        if plugin_name not in self._plugins:
            raise ValueError(f"Plugin '{plugin_name}' not found")
            
        plugin = self._plugins[plugin_name]
        
        # Remove strategies
        strategies_to_remove = [
            name for name, plugin in self._strategy_to_plugin.items()
            if plugin == plugin_name
        ]
        
        for strategy_name in strategies_to_remove:
            del self._strategies[strategy_name]
            del self._strategy_to_plugin[strategy_name]
            
        # Shutdown and remove plugin
        plugin.shutdown()
        del self._plugins[plugin_name]
        
        logger.info(
            f"Unregistered plugin '{plugin_name}' and {len(strategies_to_remove)} strategies"
        )
        
    def get_plugin(self, plugin_name: str) -> PluginInterface:
        """Get a registered plugin."""
        if plugin_name not in self._plugins:
            raise ValueError(f"Plugin '{plugin_name}' not found")
        return self._plugins[plugin_name]
        
    def get_strategy(self, strategy_name: str) -> Type[BaseStrategy]:
        """
        Get a strategy class by name.
        
        Args:
            strategy_name: Name of the strategy (can be short or full name)
            
        Returns:
            Strategy class
        """
        # Try direct lookup first
        if strategy_name in self._strategies:
            return self._strategies[strategy_name]
            
        # Try short name lookup
        matches = [
            full_name for full_name in self._strategies
            if full_name.endswith(f".{strategy_name}")
        ]
        
        if len(matches) == 1:
            return self._strategies[matches[0]]
        elif len(matches) > 1:
            raise ValueError(
                f"Ambiguous strategy name '{strategy_name}'. "
                f"Matches: {matches}"
            )
        else:
            raise ValueError(f"Strategy '{strategy_name}' not found")
            
    def list_plugins(self) -> List[Dict[str, Any]]:
        """List all registered plugins."""
        return [
            {
                "name": plugin.name,
                "version": plugin.version,
                "description": plugin.description,
                "strategies": list(plugin.get_strategies().keys())
            }
            for plugin in self._plugins.values()
        ]
        
    def list_strategies(self) -> List[Dict[str, Any]]:
        """List all available strategies."""
        strategies = []
        
        for full_name, strategy_class in self._strategies.items():
            plugin_name = self._strategy_to_plugin[full_name]
            strategy_name = full_name.split(".", 1)[1]
            
            strategies.append({
                "name": strategy_name,
                "full_name": full_name,
                "plugin": plugin_name,
                "type": getattr(strategy_class, "strategy_type", "unknown"),
                "description": strategy_class.__doc__ or "No description"
            })
            
        return strategies


class PluginLoader:
    """Loader for dynamically loading plugins."""
    
    def __init__(self, registry: PluginRegistry):
        """
        Initialize the plugin loader.
        
        Args:
            registry: Plugin registry to register loaded plugins
        """
        self.registry = registry
        
    def load_plugin_from_module(self, module_name: str) -> PluginInterface:
        """
        Load a plugin from a Python module.
        
        Args:
            module_name: Fully qualified module name
            
        Returns:
            Loaded plugin instance
        """
        try:
            module = importlib.import_module(module_name)
            
            # Find plugin class in module
            plugin_class = None
            for name, obj in inspect.getmembers(module):
                if (
                    inspect.isclass(obj) and
                    issubclass(obj, PluginInterface) and
                    obj is not PluginInterface
                ):
                    plugin_class = obj
                    break
                    
            if not plugin_class:
                raise ValueError(
                    f"No PluginInterface implementation found in {module_name}"
                )
                
            # Create and register plugin
            plugin = plugin_class()
            self.registry.register_plugin(plugin)
            
            logger.info(f"Loaded plugin from module: {module_name}")
            return plugin
            
        except Exception as e:
            logger.error(f"Failed to load plugin from {module_name}: {e}")
            raise
            
    def load_plugins_from_directory(self, directory: Path) -> List[PluginInterface]:
        """
        Load all plugins from a directory.
        
        Args:
            directory: Directory containing plugin modules
            
        Returns:
            List of loaded plugins
        """
        plugins = []
        
        if not directory.exists():
            logger.warning(f"Plugin directory does not exist: {directory}")
            return plugins
            
        # Find all Python files
        for py_file in directory.glob("*.py"):
            if py_file.name.startswith("_"):
                continue
                
            # Convert to module name
            module_name = f"{directory.name}.{py_file.stem}"
            
            try:
                plugin = self.load_plugin_from_module(module_name)
                plugins.append(plugin)
            except Exception as e:
                logger.warning(f"Failed to load plugin from {py_file}: {e}")
                
        logger.info(f"Loaded {len(plugins)} plugins from {directory}")
        return plugins
        
    def load_plugin_from_config(self, config: Dict[str, Any]) -> PluginInterface:
        """
        Load a plugin from configuration.
        
        Args:
            config: Plugin configuration with 'module' or 'path' key
            
        Returns:
            Loaded plugin instance
        """
        if "module" in config:
            plugin = self.load_plugin_from_module(config["module"])
        elif "path" in config:
            # Load from file path
            import sys
            from pathlib import Path
            
            path = Path(config["path"])
            sys.path.insert(0, str(path.parent))
            
            try:
                plugin = self.load_plugin_from_module(path.stem)
            finally:
                sys.path.pop(0)
        else:
            raise ValueError("Plugin config must have 'module' or 'path' key")
            
        # Initialize plugin with config
        if "config" in config:
            plugin.initialize(config["config"])
            
        return plugin


# Global plugin registry
_global_registry = PluginRegistry()


def get_global_registry() -> PluginRegistry:
    """Get the global plugin registry."""
    return _global_registry


def register_plugin(plugin: PluginInterface) -> None:
    """Register a plugin in the global registry."""
    _global_registry.register_plugin(plugin)


def get_strategy(strategy_name: str) -> Type[BaseStrategy]:
    """Get a strategy from the global registry."""
    return _global_registry.get_strategy(strategy_name)


class BasePlugin(PluginInterface):
    """Base implementation of a plugin for convenience."""
    
    def __init__(
        self,
        name: str,
        version: str,
        description: str,
        strategies: Optional[Dict[str, Type[BaseStrategy]]] = None
    ):
        """
        Initialize the plugin.
        
        Args:
            name: Plugin name
            version: Plugin version
            description: Plugin description
            strategies: Strategy classes provided by this plugin
        """
        self._name = name
        self._version = version
        self._description = description
        self._strategies = strategies or {}
        
    @property
    def name(self) -> str:
        return self._name
        
    @property
    def version(self) -> str:
        return self._version
        
    @property
    def description(self) -> str:
        return self._description
        
    def get_strategies(self) -> Dict[str, Type[BaseStrategy]]:
        return self._strategies
        
    def add_strategy(self, name: str, strategy_class: Type[BaseStrategy]) -> None:
        """Add a strategy to this plugin."""
        self._strategies[name] = strategy_class