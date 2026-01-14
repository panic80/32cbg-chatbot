"""Configuration loader for unified retrieval system."""

import os
import yaml
from typing import Dict, Any, Optional, List
from pathlib import Path

from app.unified_retrieval.strategies.base import PipelineConfig
from app.unified_retrieval.unified_retriever import UnifiedRetriever
from app.core.logging import get_logger
from app.core.config import settings

logger = get_logger(__name__)


class UnifiedConfigLoader:
    """Loads and manages unified retrieval configurations."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize the config loader.
        
        Args:
            config_dir: Directory containing configuration files
        """
        if config_dir:
            self.config_dir = Path(config_dir)
        else:
            # Default to the config directory in unified_retrieval module
            self.config_dir = Path(__file__).parent
            
        self._configs: Dict[str, Dict[str, Any]] = {}
        self._load_configs()
        
    def _load_configs(self) -> None:
        """Load all YAML configuration files."""
        # Load example configurations
        examples_file = self.config_dir / "unified_examples.yaml"
        if examples_file.exists():
            try:
                with open(examples_file, 'r') as f:
                    examples = yaml.safe_load(f)
                    if examples:
                        self._configs.update(examples)
                        logger.info(f"Loaded {len(examples)} example configurations")
            except Exception as e:
                logger.error(f"Failed to load example configurations: {e}")
                
        # Load custom configurations
        custom_file = self.config_dir / "custom_configs.yaml"
        if custom_file.exists():
            try:
                with open(custom_file, 'r') as f:
                    custom = yaml.safe_load(f)
                    if custom:
                        self._configs.update(custom)
                        logger.info(f"Loaded {len(custom)} custom configurations")
            except Exception as e:
                logger.error(f"Failed to load custom configurations: {e}")
                
        # Load from environment variable if specified
        env_config_path = os.getenv("UNIFIED_RETRIEVAL_CONFIG")
        if env_config_path and Path(env_config_path).exists():
            try:
                with open(env_config_path, 'r') as f:
                    env_configs = yaml.safe_load(f)
                    if env_configs:
                        self._configs.update(env_configs)
                        logger.info(f"Loaded configurations from {env_config_path}")
            except Exception as e:
                logger.error(f"Failed to load environment configurations: {e}")
                
    def get_config(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a configuration by name.
        
        Args:
            name: Configuration name
            
        Returns:
            Configuration dictionary or None if not found
        """
        return self._configs.get(name)
        
    def list_configs(self) -> List[str]:
        """Get list of available configuration names."""
        return list(self._configs.keys())
        
    def create_retriever(
        self,
        config_name: str,
        **override_params
    ) -> Optional[UnifiedRetriever]:
        """
        Create a UnifiedRetriever from a named configuration.
        
        Args:
            config_name: Name of the configuration to use
            **override_params: Parameters to override in the configuration
            
        Returns:
            Configured UnifiedRetriever or None if config not found
        """
        config = self.get_config(config_name)
        if not config:
            logger.error(f"Configuration '{config_name}' not found")
            return None
            
        # Apply overrides
        if override_params:
            config = self._apply_overrides(config, override_params)
            
        try:
            # Create pipeline config if present
            if "pipeline_config" in config:
                pipeline_config = PipelineConfig(**config["pipeline_config"])
                retriever_params = {
                    k: v for k, v in config.items()
                    if k != "pipeline_config"
                }
                retriever_params["pipeline_config"] = pipeline_config
            else:
                retriever_params = config
                
            return UnifiedRetriever(**retriever_params)
            
        except Exception as e:
            logger.error(f"Failed to create retriever from config '{config_name}': {e}")
            return None
            
    def _apply_overrides(
        self,
        config: Dict[str, Any],
        overrides: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply parameter overrides to a configuration."""
        import copy
        config = copy.deepcopy(config)
        
        # Simple overrides at top level
        for key, value in overrides.items():
            if key in config and not isinstance(config[key], dict):
                config[key] = value
                
        # Handle pipeline config overrides
        if "pipeline_config" in overrides and "pipeline_config" in config:
            for key, value in overrides["pipeline_config"].items():
                config["pipeline_config"][key] = value
                
        return config
        
    def get_default_config(self) -> str:
        """Get the default configuration name based on settings."""
        # Use settings to determine default
        mode = getattr(settings, "unified_retrieval_mode", "balanced")
        
        # Map settings mode to config name
        mode_mapping = {
            "simple": "simple",
            "balanced": "balanced",
            "advanced": "advanced",
            "table": "table_focused",
            "travel": "custom_travel_planning"
        }
        
        return mode_mapping.get(mode, "balanced")
        
    def create_default_retriever(self, **override_params) -> UnifiedRetriever:
        """Create a retriever using the default configuration."""
        default_config = self.get_default_config()
        retriever = self.create_retriever(default_config, **override_params)
        
        if not retriever:
            # Fallback to creating a simple retriever
            logger.warning(f"Failed to create default retriever, using simple fallback")
            from app.unified_retrieval.unified_retriever import UnifiedRetrieverBuilder
            from app.unified_retrieval.strategies.retrieval import VectorRetrieval
            
            retrieval_strategy = VectorRetrieval(
                name="fallback_vector_retrieval",
                search_type="similarity"
            )
            
            retriever = UnifiedRetrieverBuilder.create_simple_retriever(
                name="fallback_retriever",
                retrieval_strategy=retrieval_strategy
            )
            
        return retriever


# Global instance
_config_loader: Optional[UnifiedConfigLoader] = None


def get_config_loader() -> UnifiedConfigLoader:
    """Get or create the global config loader instance."""
    global _config_loader
    if _config_loader is None:
        _config_loader = UnifiedConfigLoader()
    return _config_loader


def load_unified_retriever(
    config_name: Optional[str] = None,
    **kwargs
) -> UnifiedRetriever:
    """
    Convenience function to load a unified retriever.
    
    Args:
        config_name: Configuration name (uses default if not specified)
        **kwargs: Override parameters
        
    Returns:
        Configured UnifiedRetriever
    """
    loader = get_config_loader()
    
    if config_name:
        return loader.create_retriever(config_name, **kwargs)
    else:
        return loader.create_default_retriever(**kwargs)