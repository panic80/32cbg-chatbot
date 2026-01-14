"""Configuration schemas for the unified retrieval system."""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, validator
from enum import Enum


class RetrieverPreset(str, Enum):
    """Pre-configured retriever types."""
    SIMPLE = "simple"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    MULTI_QUERY = "multi_query"
    CONTEXTUAL = "contextual"
    CITATION = "citation"
    CUSTOM = "custom"


class StrategyPreset(str, Enum):
    """Pre-configured strategy combinations."""
    BASIC = "basic"
    ENHANCED_QUERY = "enhanced_query"
    FILTERED = "filtered"
    SCORED = "scored"
    FULL_PIPELINE = "full_pipeline"


class VectorStoreConfig(BaseModel):
    """Configuration for vector store backends."""
    
    type: str = Field(description="Vector store type (e.g., 'chroma', 'faiss', 'pinecone')")
    collection_name: str = Field(description="Collection/index name")
    embedding_model: str = Field(default="text-embedding-ada-002", description="Embedding model")
    connection_params: Dict[str, Any] = Field(default_factory=dict, description="Connection parameters")
    search_kwargs: Dict[str, Any] = Field(default_factory=dict, description="Default search parameters")


class FilterConfig(BaseModel):
    """Configuration for document filtering."""
    
    metadata_filters: Dict[str, Any] = Field(default_factory=dict, description="Metadata field filters")
    date_range: Optional[Dict[str, str]] = Field(default=None, description="Date range filter")
    categories: Optional[List[str]] = Field(default=None, description="Category filters")
    exclude_patterns: Optional[List[str]] = Field(default=None, description="Patterns to exclude")
    custom_filter: Optional[str] = Field(default=None, description="Custom filter function name")


class ScoringConfig(BaseModel):
    """Configuration for document scoring/reranking."""
    
    method: str = Field(default="similarity", description="Scoring method")
    model: Optional[str] = Field(default=None, description="Model for scoring (if applicable)")
    weights: Dict[str, float] = Field(
        default_factory=lambda: {"similarity": 1.0},
        description="Scoring weights"
    )
    threshold: Optional[float] = Field(default=None, description="Minimum score threshold")
    normalize: bool = Field(default=True, description="Normalize scores")


class QueryEnhancementConfig(BaseModel):
    """Configuration for query enhancement strategies."""
    
    method: str = Field(default="none", description="Enhancement method")
    expand_acronyms: bool = Field(default=True, description="Expand acronyms")
    add_synonyms: bool = Field(default=False, description="Add synonyms")
    context_window: int = Field(default=0, description="Context window for enhancement")
    language_model: Optional[str] = Field(default=None, description="LLM for enhancement")
    prompt_template: Optional[str] = Field(default=None, description="Custom prompt template")


class CacheConfig(BaseModel):
    """Configuration for caching."""
    
    enabled: bool = Field(default=True, description="Enable caching")
    ttl: int = Field(default=3600, description="Cache TTL in seconds")
    max_size: int = Field(default=1000, description="Maximum cache entries")
    backend: str = Field(default="memory", description="Cache backend (memory/redis)")
    redis_url: Optional[str] = Field(default=None, description="Redis URL if using Redis backend")


class MonitoringConfig(BaseModel):
    """Configuration for monitoring and metrics."""
    
    enabled: bool = Field(default=True, description="Enable monitoring")
    log_level: str = Field(default="INFO", description="Logging level")
    metrics_interval: int = Field(default=60, description="Metrics collection interval (seconds)")
    performance_tracking: bool = Field(default=True, description="Track performance metrics")
    export_endpoint: Optional[str] = Field(default=None, description="Metrics export endpoint")


class UnifiedRetrieverConfig(BaseModel):
    """Complete configuration for a unified retriever."""
    
    # Basic settings
    name: str = Field(description="Retriever name")
    description: str = Field(default="", description="Retriever description")
    preset: RetrieverPreset = Field(default=RetrieverPreset.CUSTOM, description="Retriever preset")
    
    # Vector store configuration
    vector_stores: List[VectorStoreConfig] = Field(
        default_factory=list,
        description="Vector store configurations"
    )
    
    # Strategy configurations
    query_enhancement: QueryEnhancementConfig = Field(
        default_factory=QueryEnhancementConfig,
        description="Query enhancement configuration"
    )
    filtering: FilterConfig = Field(
        default_factory=FilterConfig,
        description="Filtering configuration"
    )
    scoring: ScoringConfig = Field(
        default_factory=ScoringConfig,
        description="Scoring configuration"
    )
    
    # Pipeline configuration
    strategy_preset: StrategyPreset = Field(
        default=StrategyPreset.BASIC,
        description="Strategy preset"
    )
    custom_strategies: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Custom strategy configurations"
    )
    parallel_groups: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Parallel execution groups"
    )
    
    # Performance settings
    top_k: int = Field(default=10, description="Default number of results")
    timeout: int = Field(default=30, description="Query timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    
    # Caching
    cache: CacheConfig = Field(
        default_factory=CacheConfig,
        description="Cache configuration"
    )
    
    # Monitoring
    monitoring: MonitoringConfig = Field(
        default_factory=MonitoringConfig,
        description="Monitoring configuration"
    )
    
    # Fallback
    fallback_retriever: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Fallback retriever configuration"
    )
    
    @validator("strategy_preset")
    def validate_strategy_preset(cls, v, values):
        """Validate strategy preset against retriever preset."""
        retriever_preset = values.get("preset", RetrieverPreset.CUSTOM)
        
        # Define valid combinations
        valid_combinations = {
            RetrieverPreset.SIMPLE: [StrategyPreset.BASIC],
            RetrieverPreset.SEMANTIC: [StrategyPreset.BASIC, StrategyPreset.ENHANCED_QUERY],
            RetrieverPreset.HYBRID: [StrategyPreset.FULL_PIPELINE],
            RetrieverPreset.MULTI_QUERY: [StrategyPreset.ENHANCED_QUERY, StrategyPreset.FULL_PIPELINE],
            RetrieverPreset.CONTEXTUAL: [StrategyPreset.FULL_PIPELINE],
            RetrieverPreset.CITATION: [StrategyPreset.FULL_PIPELINE],
            RetrieverPreset.CUSTOM: list(StrategyPreset)
        }
        
        if retriever_preset in valid_combinations:
            valid_presets = valid_combinations[retriever_preset]
            if v not in valid_presets:
                raise ValueError(
                    f"Strategy preset {v} not valid for retriever preset {retriever_preset}. "
                    f"Valid options: {valid_presets}"
                )
                
        return v
        
    def to_pipeline_config(self) -> Dict[str, Any]:
        """Convert to pipeline configuration format."""
        from ..strategies.base import StrategyConfig, PipelineConfig
        
        strategies = []
        
        # Build strategies based on preset
        if self.strategy_preset == StrategyPreset.BASIC:
            # Just retrieval
            strategies.append({
                "strategy_class": "app.unified_retrieval.strategies.retrieval.VectorStoreStrategy",
                "strategy_type": "retrieval",
                "config": {
                    "vector_stores": [vs.dict() for vs in self.vector_stores],
                    "top_k": self.top_k
                }
            })
            
        elif self.strategy_preset == StrategyPreset.ENHANCED_QUERY:
            # Query enhancement + retrieval
            strategies.extend([
                {
                    "strategy_class": "app.unified_retrieval.strategies.query.QueryEnhancementStrategy",
                    "strategy_type": "query_enhancement",
                    "config": self.query_enhancement.dict()
                },
                {
                    "strategy_class": "app.unified_retrieval.strategies.retrieval.VectorStoreStrategy",
                    "strategy_type": "retrieval",
                    "config": {
                        "vector_stores": [vs.dict() for vs in self.vector_stores],
                        "top_k": self.top_k
                    }
                }
            ])
            
        elif self.strategy_preset == StrategyPreset.FILTERED:
            # Retrieval + filtering
            strategies.extend([
                {
                    "strategy_class": "app.unified_retrieval.strategies.retrieval.VectorStoreStrategy",
                    "strategy_type": "retrieval",
                    "config": {
                        "vector_stores": [vs.dict() for vs in self.vector_stores],
                        "top_k": self.top_k * 2  # Get more for filtering
                    }
                },
                {
                    "strategy_class": "app.unified_retrieval.strategies.filtering.MetadataFilterStrategy",
                    "strategy_type": "filtering",
                    "config": self.filtering.dict()
                }
            ])
            
        elif self.strategy_preset == StrategyPreset.SCORED:
            # Retrieval + scoring
            strategies.extend([
                {
                    "strategy_class": "app.unified_retrieval.strategies.retrieval.VectorStoreStrategy",
                    "strategy_type": "retrieval",
                    "config": {
                        "vector_stores": [vs.dict() for vs in self.vector_stores],
                        "top_k": self.top_k * 2  # Get more for scoring
                    }
                },
                {
                    "strategy_class": "app.unified_retrieval.strategies.scoring.ReRankingStrategy",
                    "strategy_type": "scoring",
                    "config": self.scoring.dict()
                }
            ])
            
        elif self.strategy_preset == StrategyPreset.FULL_PIPELINE:
            # Full pipeline: enhancement -> retrieval -> filtering -> scoring
            strategies.extend([
                {
                    "strategy_class": "app.unified_retrieval.strategies.query.QueryEnhancementStrategy",
                    "strategy_type": "query_enhancement",
                    "config": self.query_enhancement.dict(),
                    "order": 1
                },
                {
                    "strategy_class": "app.unified_retrieval.strategies.retrieval.VectorStoreStrategy",
                    "strategy_type": "retrieval",
                    "config": {
                        "vector_stores": [vs.dict() for vs in self.vector_stores],
                        "top_k": self.top_k * 3  # Get more for filtering and scoring
                    },
                    "order": 2
                },
                {
                    "strategy_class": "app.unified_retrieval.strategies.filtering.MetadataFilterStrategy",
                    "strategy_type": "filtering",
                    "config": self.filtering.dict(),
                    "order": 3
                },
                {
                    "strategy_class": "app.unified_retrieval.strategies.scoring.ReRankingStrategy",
                    "strategy_type": "scoring",
                    "config": self.scoring.dict(),
                    "order": 4
                }
            ])
            
        # Add custom strategies
        strategies.extend(self.custom_strategies)
        
        # Apply parallel groups
        for strategy in strategies:
            for group_name, strategy_names in self.parallel_groups.items():
                if strategy.get("strategy_class", "").split(".")[-1] in strategy_names:
                    strategy["parallel_group"] = group_name
                    
        # Create pipeline config
        pipeline_config = {
            "name": f"{self.name}_pipeline",
            "description": self.description,
            "strategies": strategies,
            "config": {
                "timeout": self.timeout,
                "max_retries": self.max_retries,
                "deduplicate_docs": True
            }
        }
        
        return pipeline_config
        
    def to_retriever_config(self) -> Dict[str, Any]:
        """Convert to retriever configuration format."""
        config = {
            "name": self.name,
            "description": self.description,
            "enable_caching": self.cache.enabled,
            "cache_ttl": self.cache.ttl,
            "pipeline": self.to_pipeline_config()
        }
        
        if self.fallback_retriever:
            config["fallback_retriever_config"] = self.fallback_retriever
            
        return config


class RetrieverConfigBuilder:
    """Builder for creating retriever configurations."""
    
    @staticmethod
    def build_simple_semantic_retriever(
        name: str,
        collection_name: str,
        embedding_model: str = "text-embedding-ada-002",
        **kwargs
    ) -> UnifiedRetrieverConfig:
        """Build configuration for a simple semantic retriever."""
        return UnifiedRetrieverConfig(
            name=name,
            description="Simple semantic search retriever",
            preset=RetrieverPreset.SEMANTIC,
            strategy_preset=StrategyPreset.BASIC,
            vector_stores=[
                VectorStoreConfig(
                    type="chroma",
                    collection_name=collection_name,
                    embedding_model=embedding_model
                )
            ],
            **kwargs
        )
        
    @staticmethod
    def build_hybrid_retriever(
        name: str,
        vector_collection: str,
        bm25_index: str,
        **kwargs
    ) -> UnifiedRetrieverConfig:
        """Build configuration for a hybrid retriever."""
        return UnifiedRetrieverConfig(
            name=name,
            description="Hybrid semantic + keyword search retriever",
            preset=RetrieverPreset.HYBRID,
            strategy_preset=StrategyPreset.FULL_PIPELINE,
            vector_stores=[
                VectorStoreConfig(
                    type="chroma",
                    collection_name=vector_collection
                )
            ],
            custom_strategies=[
                {
                    "strategy_class": "app.unified_retrieval.strategies.retrieval.BM25Strategy",
                    "strategy_type": "retrieval",
                    "config": {"index_name": bm25_index},
                    "parallel_group": "hybrid_retrieval",
                    "order": 2
                }
            ],
            parallel_groups={
                "hybrid_retrieval": ["VectorStoreStrategy", "BM25Strategy"]
            },
            **kwargs
        )
        
    @staticmethod
    def build_citation_retriever(
        name: str,
        collection_name: str,
        **kwargs
    ) -> UnifiedRetrieverConfig:
        """Build configuration for a citation-aware retriever."""
        return UnifiedRetrieverConfig(
            name=name,
            description="Citation-aware retriever with source tracking",
            preset=RetrieverPreset.CITATION,
            strategy_preset=StrategyPreset.FULL_PIPELINE,
            vector_stores=[
                VectorStoreConfig(
                    type="chroma",
                    collection_name=collection_name
                )
            ],
            scoring=ScoringConfig(
                method="citation_aware",
                weights={
                    "similarity": 0.7,
                    "citation_quality": 0.3
                }
            ),
            custom_strategies=[
                {
                    "strategy_class": "app.unified_retrieval.strategies.post_processing.CitationEnrichmentStrategy",
                    "strategy_type": "post_processing",
                    "config": {"add_citations": True},
                    "order": 5
                }
            ],
            **kwargs
        )