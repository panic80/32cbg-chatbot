"""Unified retrieval strategies."""

from app.unified_retrieval.strategies.base import (
    BaseStrategy,
    RetrievalContext,
    StrategyType,
    StrategyPipeline,
    StrategyConfig,
    PipelineConfig
)

# Query Enhancement Strategies
from app.unified_retrieval.strategies.query_enhancement import (
    MultiQueryStrategy,
    SelfQueryStrategy,
    QueryExpansionStrategy,
    AbbreviationExpansionStrategy
)

# Filtering Strategies
from app.unified_retrieval.strategies.filtering import (
    ContextAwareFilterStrategy,
    RestrictionAwareFilterStrategy,
    ClassAFilterStrategy,
    MetadataFilterStrategy
)

# Scoring Strategies
from app.unified_retrieval.strategies.scoring import (
    ContentBoostStrategy,
    AuthorityBoostStrategy,
    CooccurrenceScoreStrategy,
    HybridScoreStrategy
)

# Retrieval Strategies
from app.unified_retrieval.strategies.retrieval import (
    VectorRetrievalStrategy,
    BM25RetrievalStrategy,
    HybridRetrievalStrategy,
    ParentDocumentStrategy
)

__all__ = [
    # Base classes
    "BaseStrategy",
    "RetrievalContext",
    "StrategyType",
    "StrategyPipeline",
    "StrategyConfig",
    "PipelineConfig",
    
    # Query Enhancement
    "MultiQueryStrategy",
    "SelfQueryStrategy",
    "QueryExpansionStrategy",
    "AbbreviationExpansionStrategy",
    
    # Filtering
    "ContextAwareFilterStrategy",
    "RestrictionAwareFilterStrategy",
    "ClassAFilterStrategy",
    "MetadataFilterStrategy",
    
    # Scoring
    "ContentBoostStrategy",
    "AuthorityBoostStrategy",
    "CooccurrenceScoreStrategy",
    "HybridScoreStrategy",
    
    # Retrieval
    "VectorRetrievalStrategy",
    "BM25RetrievalStrategy",
    "HybridRetrievalStrategy",
    "ParentDocumentStrategy",
    
    # Enhanced Strategies (temporarily disabled)
    # "SemanticCacheStrategy",
    # "EnhancedQueryStrategy",
    # "ContextualBoostStrategy",
    # "RRFHybridStrategy",
    # "AdaptiveHybridStrategy",
    # "ComprehensiveEnhancementStrategy"
]

# Import enhanced strategies
# NOTE: Temporarily disabled due to missing spacy dependency
# from app.unified_retrieval.strategies.enhanced_strategies import (
#     SemanticCacheStrategy,
#     EnhancedQueryStrategy,
#     ContextualBoostStrategy,
#     RRFHybridStrategy,
#     AdaptiveHybridStrategy,
#     ComprehensiveEnhancementStrategy
# )