"""Unified retrieval framework for RAG systems."""

from .unified_retriever import (
    UnifiedRetriever,
    UnifiedRetrieverBuilder
)
from .strategies.base import (
    BaseStrategy,
    StrategyType,
    RetrievalContext,
    StrategyPipeline,
    PipelineConfig,
    StrategyConfig
)
from .config.schemas import (
    UnifiedRetrieverConfig,
    RetrieverPreset,
    StrategyPreset,
    VectorStoreConfig,
    FilterConfig,
    ScoringConfig,
    QueryEnhancementConfig,
    CacheConfig,
    MonitoringConfig,
    RetrieverConfigBuilder
)
from .plugins.base import (
    PluginInterface,
    PluginRegistry,
    PluginLoader,
    BasePlugin,
    get_global_registry,
    register_plugin,
    get_strategy
)
from .migration import (
    LegacyConfigMapper,
    validate_unified_config,
    create_example_unified_config
)
from .config.loader import (
    UnifiedConfigLoader,
    get_config_loader,
    load_unified_retriever
)

__all__ = [
    # Main retriever
    "UnifiedRetriever",
    "UnifiedRetrieverBuilder",
    
    # Strategy system
    "BaseStrategy",
    "StrategyType",
    "RetrievalContext",
    "StrategyPipeline",
    "PipelineConfig",
    "StrategyConfig",
    
    # Configuration
    "UnifiedRetrieverConfig",
    "RetrieverPreset",
    "StrategyPreset",
    "VectorStoreConfig",
    "FilterConfig",
    "ScoringConfig",
    "QueryEnhancementConfig",
    "CacheConfig",
    "MonitoringConfig",
    "RetrieverConfigBuilder",
    
    # Plugin system
    "PluginInterface",
    "PluginRegistry",
    "PluginLoader",
    "BasePlugin",
    "get_global_registry",
    "register_plugin",
    "get_strategy",
    
    # Migration utilities
    "LegacyConfigMapper",
    "validate_unified_config",
    "create_example_unified_config",
    
    # Config loading
    "UnifiedConfigLoader",
    "get_config_loader",
    "load_unified_retriever"
]