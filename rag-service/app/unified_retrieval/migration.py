"""Migration utilities for converting legacy configurations to unified retrieval format."""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass

from langchain_core.language_models import BaseLLM
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from app.unified_retrieval.unified_retriever import UnifiedRetriever, UnifiedRetrieverBuilder
from app.unified_retrieval.strategies.base import (
    StrategyType, PipelineConfig, StrategyConfig, StrategyPipeline
)
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class StrategyMapping:
    """Maps legacy component types to unified strategies."""
    legacy_type: str
    strategy_class: str
    strategy_type: StrategyType
    default_config: Dict[str, Any]


# Define mappings from legacy components to unified strategies
STRATEGY_MAPPINGS = [
    # Query enhancement strategies
    StrategyMapping(
        legacy_type="multi_query",
        strategy_class="app.unified_retrieval.strategies.query_enhancement.MultiQueryStrategy",
        strategy_type=StrategyType.QUERY_ENHANCEMENT,
        default_config={"num_queries": 3, "include_original": True}
    ),
    StrategyMapping(
        legacy_type="self_query",
        strategy_class="app.unified_retrieval.strategies.query_enhancement.SelfQueryEnhancer",
        strategy_type=StrategyType.QUERY_ENHANCEMENT,
        default_config={"document_contents": "Canadian Forces travel instructions"}
    ),
    
    # Retrieval strategies
    StrategyMapping(
        legacy_type="vector_similarity",
        strategy_class="app.unified_retrieval.strategies.retrieval.VectorRetrievalStrategy",
        strategy_type=StrategyType.RETRIEVAL,
        default_config={"search_type": "similarity"}
    ),
    StrategyMapping(
        legacy_type="vector_mmr",
        strategy_class="app.unified_retrieval.strategies.retrieval.VectorRetrievalStrategy",
        strategy_type=StrategyType.RETRIEVAL,
        default_config={"search_type": "mmr", "lambda_mult": 0.5}
    ),
    StrategyMapping(
        legacy_type="bm25",
        strategy_class="app.unified_retrieval.strategies.retrieval.BM25RetrievalStrategy",
        strategy_type=StrategyType.RETRIEVAL,
        default_config={}
    ),
    StrategyMapping(
        legacy_type="parent_document",
        strategy_class="app.unified_retrieval.strategies.retrieval.ParentDocumentRetrieval",
        strategy_type=StrategyType.RETRIEVAL,
        default_config={}
    ),
    StrategyMapping(
        legacy_type="cooccurrence",
        strategy_class="app.unified_retrieval.strategies.retrieval.CooccurrenceRetrieval",
        strategy_type=StrategyType.RETRIEVAL,
        default_config={}
    ),
    
    # Scoring/reranking strategies
    StrategyMapping(
        legacy_type="authority_reranker",
        strategy_class="app.unified_retrieval.strategies.scoring.AuthorityScoring",
        strategy_type=StrategyType.RERANKING,
        default_config={"boost_factor": 2.0}
    ),
    StrategyMapping(
        legacy_type="contextual_compressor",
        strategy_class="app.unified_retrieval.strategies.scoring.ContextualCompression",
        strategy_type=StrategyType.POST_PROCESSING,
        default_config={"compression_mode": "hybrid"}
    ),
]


class LegacyConfigMapper:
    """Maps legacy retriever configurations to unified retrieval format."""
    
    def __init__(
        self,
        vectorstore: Optional[VectorStore] = None,
        llm: Optional[BaseLLM] = None,
        embeddings: Optional[Embeddings] = None,
        all_documents: Optional[List] = None
    ):
        """
        Initialize the mapper with required components.
        
        Args:
            vectorstore: Vector store for similarity search
            llm: Language model for advanced features
            embeddings: Embeddings for compression
            all_documents: Documents for BM25 retrieval
        """
        self.vectorstore = vectorstore
        self.llm = llm
        self.embeddings = embeddings
        self.all_documents = all_documents
        self._mapping_dict = {m.legacy_type: m for m in STRATEGY_MAPPINGS}
        
    def map_retriever_config(self, config: Any) -> UnifiedRetriever:
        """
        Map a legacy RetrieverConfig to a UnifiedRetriever.
        
        Args:
            config: Legacy RetrieverConfig instance
            
        Returns:
            Configured UnifiedRetriever
        """
        # Determine which strategies to include based on config
        strategies = []
        
        # Query enhancement strategies
        # Multi-query disabled for performance (saves ~6s, only 0.1 RRF weight)
        # if config.use_multi_query and self.llm:
        #     strategies.append(self._create_strategy_config("multi_query", order=10))
            
        if config.use_self_query and self.llm:
            strategies.append(self._create_strategy_config("self_query", order=20))
            
        # Retrieval strategies (parallel group)
        retrieval_strategies = []
        
        # Always add vector similarity
        retrieval_strategies.append(
            self._create_strategy_config("vector_similarity", parallel_group="retrieval")
        )
        
        # Add MMR for diversity
        retrieval_strategies.append(
            self._create_strategy_config("vector_mmr", parallel_group="retrieval")
        )
        
        # Add BM25 if enabled
        if config.use_bm25 and self.all_documents:
            retrieval_strategies.append(
                self._create_strategy_config("bm25", parallel_group="retrieval")
            )
            
        # Add smart chunking (parent document retriever)
        if config.use_smart_chunking:
            retrieval_strategies.append(
                self._create_strategy_config("parent_document", parallel_group="retrieval")
            )
            
        # Add cooccurrence if enabled
        if config.use_cooccurrence and self.all_documents:
            retrieval_strategies.append(
                self._create_strategy_config("cooccurrence", parallel_group="retrieval")
            )
            
        strategies.extend(retrieval_strategies)
        
        # Post-processing strategies
        if config.use_reranking:
            strategies.append(self._create_strategy_config("authority_reranker", order=100))
            
        if config.use_compression and self.llm and self.embeddings:
            strategies.append(self._create_strategy_config("contextual_compressor", order=110))
            
        # Create pipeline configuration
        pipeline_config = PipelineConfig(
            name=f"legacy_mapped_{config.mode.value}",
            description=f"Pipeline mapped from legacy {config.mode.value} mode",
            strategies=strategies,
            config={
                "k": config.k,
                "ensemble_weights": config.ensemble_weights,
                "deduplicate_docs": True
            }
        )
        
        # Create unified retriever
        retriever = UnifiedRetriever(
            name=f"mapped_{config.mode.value}_retriever",
            pipeline_config=pipeline_config,
            enable_caching=True
        )
        
        # Store component references for strategies to use
        retriever.vectorstore = self.vectorstore
        retriever.llm = self.llm
        retriever.embeddings = self.embeddings
        retriever.all_documents = self.all_documents
        
        return retriever
        
    def _create_strategy_config(
        self,
        legacy_type: str,
        order: Optional[int] = None,
        parallel_group: Optional[str] = None
    ) -> StrategyConfig:
        """Create a strategy configuration from legacy type."""
        mapping = self._mapping_dict.get(legacy_type)
        if not mapping:
            raise ValueError(f"Unknown legacy type: {legacy_type}")
            
        return StrategyConfig(
            strategy_class=mapping.strategy_class,
            strategy_type=mapping.strategy_type,
            enabled=True,
            config=mapping.default_config.copy(),
            order=order,
            parallel_group=parallel_group
        )
        
    def map_retriever_dict(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map a dictionary retriever configuration to unified format.
        
        Args:
            config_dict: Dictionary with retriever configuration
            
        Returns:
            Dictionary configuration for UnifiedRetriever
        """
        retriever_type = config_dict.get("type", "vector")
        
        if retriever_type == "vector":
            search_type = config_dict.get("search_type", "similarity")
            if search_type == "similarity":
                strategy_class = "app.unified_retrieval.strategies.retrieval.VectorRetrievalStrategy"
                strategy_config = {"search_type": "similarity", "k": config_dict.get("k", 10)}
            elif search_type == "mmr":
                strategy_class = "app.unified_retrieval.strategies.retrieval.VectorRetrievalStrategy"
                strategy_config = {
                    "search_type": "mmr",
                    "k": config_dict.get("k", 10),
                    "lambda_mult": config_dict.get("lambda_mult", 0.5)
                }
            else:
                raise ValueError(f"Unknown vector search type: {search_type}")
                
        elif retriever_type == "bm25":
            strategy_class = "app.unified_retrieval.strategies.retrieval.BM25RetrievalStrategy"
            strategy_config = {"k": config_dict.get("k", 10)}
            
        elif retriever_type == "multi_query":
            # This needs to be handled as a pipeline with query enhancement + retrieval
            base_config = config_dict.get("base_retriever", {"type": "vector"})
            base_strategy = self.map_retriever_dict(base_config)
            
            return {
                "pipeline_config": {
                    "name": "multi_query_pipeline",
                    "strategies": [
                        {
                            "strategy_class": "app.unified_retrieval.strategies.query_enhancement.MultiQueryStrategy",
                            "strategy_type": "query_enhancement",
                            "config": {"num_queries": 3, "include_original": True},
                            "order": 1
                        },
                        base_strategy["pipeline_config"]["strategies"][0]
                    ]
                }
            }
        else:
            raise ValueError(f"Unknown retriever type: {retriever_type}")
            
        return {
            "pipeline_config": {
                "name": f"{retriever_type}_pipeline",
                "strategies": [
                    {
                        "strategy_class": strategy_class,
                        "strategy_type": "retrieval",
                        "config": strategy_config,
                        "order": 1
                    }
                ]
            }
        }


def validate_unified_config(config: Dict[str, Any]) -> bool:
    """
    Validate a unified retriever configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Check for required fields
    if "pipeline_config" not in config and "strategies" not in config:
        raise ValueError("Configuration must include either 'pipeline_config' or 'strategies'")
        
    # Validate pipeline config if present
    if "pipeline_config" in config:
        pipeline_config = config["pipeline_config"]
        if not isinstance(pipeline_config.get("strategies", []), list):
            raise ValueError("Pipeline config must include a list of strategies")
            
        for i, strategy in enumerate(pipeline_config["strategies"]):
            if "strategy_class" not in strategy:
                raise ValueError(f"Strategy {i} missing required field 'strategy_class'")
            if "strategy_type" not in strategy:
                raise ValueError(f"Strategy {i} missing required field 'strategy_type'")
                
    return True


def create_example_unified_config(mode: str = "balanced") -> Dict[str, Any]:
    """
    Create an example unified retriever configuration.
    
    Args:
        mode: Configuration mode ("simple", "balanced", "advanced")
        
    Returns:
        Configuration dictionary
    """
    if mode == "simple":
        return {
            "name": "simple_unified_retriever",
            "pipeline_config": {
                "name": "simple_pipeline",
                "description": "Simple vector similarity search",
                "strategies": [
                    {
                        "strategy_class": "app.unified_retrieval.strategies.retrieval.VectorRetrievalStrategy",
                        "strategy_type": "retrieval",
                        "config": {"search_type": "similarity", "k": 10},
                        "order": 1
                    }
                ]
            }
        }
        
    elif mode == "balanced":
        return {
            "name": "balanced_unified_retriever",
            "pipeline_config": {
                "name": "balanced_pipeline",
                "description": "Balanced retrieval with multiple strategies",
                "strategies": [
                    # Query enhancement
                    {
                        "strategy_class": "app.unified_retrieval.strategies.query_enhancement.MultiQueryStrategy",
                        "strategy_type": "query_enhancement",
                        "config": {"num_queries": 3},
                        "order": 1
                    },
                    # Parallel retrieval strategies
                    {
                        "strategy_class": "app.unified_retrieval.strategies.retrieval.VectorRetrievalStrategy",
                        "strategy_type": "retrieval",
                        "config": {"search_type": "similarity"},
                        "parallel_group": "retrieval"
                    },
                    {
                        "strategy_class": "app.unified_retrieval.strategies.retrieval.VectorRetrievalStrategy",
                        "strategy_type": "retrieval",
                        "config": {"search_type": "mmr", "lambda_mult": 0.5},
                        "parallel_group": "retrieval"
                    },
                    {
                        "strategy_class": "app.unified_retrieval.strategies.retrieval.BM25RetrievalStrategy",
                        "strategy_type": "retrieval",
                        "config": {},
                        "parallel_group": "retrieval"
                    },
                    # Scoring/reranking
                    {
                        "strategy_class": "app.unified_retrieval.strategies.scoring.HybridScoreStrategy",
                        "strategy_type": "scoring",
                        "config": {"weights": {"similarity": 0.4, "mmr": 0.3, "bm25": 0.3}},
                        "order": 100
                    }
                ]
            }
        }
        
    elif mode == "advanced":
        return {
            "name": "advanced_unified_retriever",
            "enable_caching": True,
            "cache_ttl": 3600,
            "pipeline_config": {
                "name": "advanced_pipeline",
                "description": "Advanced multi-stage retrieval pipeline",
                "strategies": [
                    # Query understanding and enhancement
                    {
                        "strategy_class": "app.unified_retrieval.strategies.query_enhancement.QueryClassifier",
                        "strategy_type": "query_enhancement",
                        "config": {},
                        "order": 1
                    },
                    {
                        "strategy_class": "app.unified_retrieval.strategies.query_enhancement.MultiQueryStrategy",
                        "strategy_type": "query_enhancement",
                        "config": {"num_queries": 5, "include_original": True},
                        "order": 2
                    },
                    # Filtering
                    {
                        "strategy_class": "app.unified_retrieval.strategies.filtering.MetadataFilter",
                        "strategy_type": "filtering",
                        "config": {"filters": {}},
                        "order": 3
                    },
                    # Parallel retrieval
                    {
                        "strategy_class": "app.unified_retrieval.strategies.retrieval.VectorRetrievalStrategy",
                        "strategy_type": "retrieval",
                        "config": {"search_type": "similarity"},
                        "parallel_group": "retrieval"
                    },
                    {
                        "strategy_class": "app.unified_retrieval.strategies.retrieval.VectorRetrievalStrategy",
                        "strategy_type": "retrieval",
                        "config": {"search_type": "mmr"},
                        "parallel_group": "retrieval"
                    },
                    {
                        "strategy_class": "app.unified_retrieval.strategies.retrieval.BM25RetrievalStrategy",
                        "strategy_type": "retrieval",
                        "config": {},
                        "parallel_group": "retrieval"
                    },
                    {
                        "strategy_class": "app.unified_retrieval.strategies.retrieval.ParentDocumentRetrieval",
                        "strategy_type": "retrieval",
                        "config": {},
                        "parallel_group": "retrieval"
                    },
                    # Scoring and reranking
                    {
                        "strategy_class": "app.unified_retrieval.strategies.scoring.HybridScoreStrategy",
                        "strategy_type": "scoring",
                        "config": {"merge_strategy": "weighted"},
                        "order": 100
                    },
                    {
                        "strategy_class": "app.unified_retrieval.strategies.scoring.AuthorityScoring",
                        "strategy_type": "reranking",
                        "config": {"boost_factor": 2.0},
                        "order": 110
                    },
                    # Post-processing
                    {
                        "strategy_class": "app.unified_retrieval.strategies.scoring.ContextualCompression",
                        "strategy_type": "post_processing",
                        "config": {"compression_mode": "hybrid"},
                        "order": 120
                    }
                ],
                "config": {
                    "k": 15,
                    "deduplicate_docs": True
                }
            }
        }
        
    else:
        raise ValueError(f"Unknown mode: {mode}")