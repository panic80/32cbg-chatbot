"""Enhanced Retrieval Strategies with Advanced Quality Enhancements

This module provides strategy implementations that leverage the advanced
retrieval quality enhancements including semantic caching, query expansion,
contextual retrieval, and enhanced hybrid search.
"""

from typing import Any, Dict, List, Optional
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.embeddings import Embeddings
import redis
import logging

from ..enhancements.semantic_cache import SemanticCache
from ..enhancements.query_expansion import AdvancedQueryExpander
from ..enhancements.contextual_retrieval import ContextualRetriever
from ..enhancements.hybrid_search import EnhancedHybridSearch
from .base import BaseStrategy, RetrievalContext

logger = logging.getLogger(__name__)


class SemanticCacheStrategy(BaseStrategy):
    """Strategy that checks semantic cache before retrieval"""
    
    def __init__(
        self,
        embeddings: Embeddings,
        redis_client: Optional[redis.Redis] = None,
        similarity_threshold: float = 0.95,
        ttl: int = 3600
    ):
        """Initialize semantic cache strategy
        
        Args:
            embeddings: Embeddings model
            redis_client: Redis client for cache storage
            similarity_threshold: Minimum similarity for cache hit
            ttl: Cache TTL in seconds
        """
        super().__init__()
        self.cache = SemanticCache(
            embeddings=embeddings,
            redis_client=redis_client,
            similarity_threshold=similarity_threshold,
            ttl=ttl
        )
    
    async def execute(
        self, 
        context: RetrievalContext, 
        **kwargs
    ) -> RetrievalContext:
        """Execute semantic cache check"""
        # Check cache for similar query
        cache_result = await self.cache.get(
            context.query,
            context={"conversation_id": kwargs.get("conversation_id")}
        )
        
        if cache_result:
            documents, metadata = cache_result
            context.documents = documents
            context.metadata.update(metadata)
            context.metadata["cache_strategy"] = "hit"
            
            logger.info(f"Semantic cache hit for query: {context.query}")
            
            # Skip further processing if cache hit
            context.metadata["skip_retrieval"] = True
        else:
            context.metadata["cache_strategy"] = "miss"
            
            # Store result in cache after retrieval (in post-processing)
            context.metadata["cache_query"] = True
        
        return context
    
    async def post_process(self, context: RetrievalContext, **kwargs) -> None:
        """Cache results after retrieval"""
        if context.metadata.get("cache_query") and context.documents:
            await self.cache.set(
                context.query,
                context.documents,
                metadata=context.metadata,
                context={"conversation_id": kwargs.get("conversation_id")}
            )


class EnhancedQueryStrategy(BaseStrategy):
    """Strategy that applies advanced query expansion"""
    
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """Initialize query expansion strategy
        
        Args:
            spacy_model: SpaCy model for NLP
        """
        super().__init__()
        self.expander = AdvancedQueryExpander(spacy_model=spacy_model)
    
    async def execute(
        self,
        context: RetrievalContext,
        **kwargs
    ) -> RetrievalContext:
        """Execute query expansion"""
        # Get conversation history if available
        conversation_history = kwargs.get("conversation_history", [])
        
        # Expand query
        expansion_result = self.expander.expand_query(
            context.query,
            conversation_history=conversation_history,
            context=kwargs.get("context", {})
        )
        
        # Update context
        context.enhanced_query = expansion_result["expanded_query"]
        context.metadata["query_expansion"] = {
            "original": expansion_result["original_query"],
            "expanded": expansion_result["expanded_query"],
            "terms_added": expansion_result["expanded_terms"],
            "entities": expansion_result["entities"],
            "temporal_context": expansion_result["temporal_context"],
            "abbreviations_expanded": expansion_result["abbreviations_expanded"]
        }
        
        # Determine query type for optimization
        query_type = self.expander.get_query_type(context.query)
        context.metadata["query_type"] = query_type
        
        logger.info(f"Query expanded: {context.query} -> {context.enhanced_query}")
        
        return context


class ContextualBoostStrategy(BaseStrategy):
    """Strategy that applies contextual scoring based on conversation history"""
    
    def __init__(
        self,
        embeddings: Embeddings,
        memory_size: int = 10,
        entity_boost: float = 0.2,
        topic_boost: float = 0.15,
        continuity_boost: float = 0.1
    ):
        """Initialize contextual boost strategy
        
        Args:
            embeddings: Embeddings model
            memory_size: Conversation memory size
            entity_boost: Entity match score boost
            topic_boost: Topic relevance score boost
            continuity_boost: Topic continuity score boost
        """
        super().__init__()
        self.contextual_retriever = ContextualRetriever(
            embeddings=embeddings,
            memory_size=memory_size,
            entity_boost=entity_boost,
            topic_boost=topic_boost,
            continuity_boost=continuity_boost
        )
    
    async def execute(
        self,
        context: RetrievalContext,
        **kwargs
    ) -> RetrievalContext:
        """Apply contextual scoring to documents"""
        if not context.documents:
            return context
        
        # Get base scores if available
        scores = context.metadata.get("scores", [1.0] * len(context.documents))
        
        # Apply contextual enhancement
        enhanced_docs, enhanced_scores = self.contextual_retriever.enhance_retrieval(
            context.query,
            context.documents,
            scores
        )
        
        # Update context
        context.documents = enhanced_docs
        context.metadata["scores"] = enhanced_scores
        context.metadata["contextual_enhancement"] = True
        
        # Get context summary for metadata
        context_summary = self.contextual_retriever.memory.get_context_summary()
        context.metadata["context_summary"] = context_summary
        
        return context
    
    async def post_process(self, context: RetrievalContext, **kwargs) -> None:
        """Update conversation memory after retrieval"""
        if context.documents and kwargs.get("response"):
            self.contextual_retriever.update_context(
                context.query,
                kwargs["response"],
                context.documents
            )


class RRFHybridStrategy(BaseStrategy):
    """Strategy using Reciprocal Rank Fusion for hybrid search"""
    
    def __init__(
        self,
        rrf_k: int = 60,
        normalization_method: str = "minmax"
    ):
        """Initialize RRF hybrid strategy
        
        Args:
            rrf_k: RRF parameter
            normalization_method: Score normalization method
        """
        super().__init__()
        self.hybrid_search = EnhancedHybridSearch(
            rrf_k=rrf_k,
            normalization_method=normalization_method,
            fusion_strategy="rrf"
        )
    
    async def execute(
        self,
        context: RetrievalContext,
        **kwargs
    ) -> RetrievalContext:
        """Execute RRF fusion on hybrid results"""
        # Check if we have both dense and sparse results
        dense_results = context.metadata.get("dense_results", [])
        sparse_results = context.metadata.get("sparse_results", [])
        
        if not (dense_results or sparse_results):
            return context
        
        # Determine query type for optimal weights
        query_type = context.metadata.get("query_type", "hybrid")
        
        # Get fusion configuration
        fusion_config = self.hybrid_search.get_fusion_config(query_type)
        
        # Apply RRF fusion
        fused_results = self.hybrid_search.fuse_results(
            dense_results,
            sparse_results,
            query_type=query_type,
            strategy=fusion_config["strategy"]
        )
        
        # Update context
        context.documents = [doc for doc, _ in fused_results]
        context.metadata["scores"] = [score for _, score in fused_results]
        context.metadata["fusion_strategy"] = fusion_config["strategy"]
        context.metadata["fusion_config"] = fusion_config
        
        logger.info(f"Applied RRF fusion with config: {fusion_config}")
        
        return context


class AdaptiveHybridStrategy(BaseStrategy):
    """Strategy that dynamically adjusts hybrid search based on query analysis"""
    
    def __init__(self):
        """Initialize adaptive hybrid strategy"""
        super().__init__()
        self.hybrid_search = EnhancedHybridSearch()
    
    async def execute(
        self,
        context: RetrievalContext,
        **kwargs
    ) -> RetrievalContext:
        """Execute adaptive hybrid fusion"""
        # Analyze query type
        query_type = self.hybrid_search.analyze_query_type(context.query)
        context.metadata["detected_query_type"] = query_type
        
        # Get optimal fusion configuration
        fusion_config = self.hybrid_search.get_fusion_config(query_type)
        
        # Update hybrid search parameters
        self.hybrid_search.dense_weight = fusion_config.get("dense_weight", 0.5)
        self.hybrid_search.sparse_weight = fusion_config.get("sparse_weight", 0.5)
        self.hybrid_search.fusion_strategy = fusion_config.get("strategy", "rrf")
        
        # Get dense and sparse results
        dense_results = context.metadata.get("dense_results", [])
        sparse_results = context.metadata.get("sparse_results", [])
        
        if dense_results or sparse_results:
            # Apply adaptive fusion
            fused_results = self.hybrid_search.fuse_results(
                dense_results,
                sparse_results,
                query_type=query_type
            )
            
            # Update context
            context.documents = [doc for doc, _ in fused_results]
            context.metadata["scores"] = [score for _, score in fused_results]
            context.metadata["adaptive_fusion"] = True
            context.metadata["fusion_config"] = fusion_config
        
        return context


class ComprehensiveEnhancementStrategy(BaseStrategy):
    """Comprehensive strategy combining all enhancements"""
    
    def __init__(
        self,
        embeddings: Embeddings,
        redis_client: Optional[redis.Redis] = None,
        enable_cache: bool = True,
        enable_expansion: bool = True,
        enable_contextual: bool = True,
        enable_hybrid: bool = True
    ):
        """Initialize comprehensive enhancement strategy
        
        Args:
            embeddings: Embeddings model
            redis_client: Redis client
            enable_cache: Enable semantic caching
            enable_expansion: Enable query expansion
            enable_contextual: Enable contextual scoring
            enable_hybrid: Enable enhanced hybrid search
        """
        super().__init__()
        
        # Initialize sub-strategies
        if enable_cache:
            self.cache_strategy = SemanticCacheStrategy(embeddings, redis_client)
        else:
            self.cache_strategy = None
            
        if enable_expansion:
            self.expansion_strategy = EnhancedQueryStrategy()
        else:
            self.expansion_strategy = None
            
        if enable_contextual:
            self.contextual_strategy = ContextualBoostStrategy(embeddings)
        else:
            self.contextual_strategy = None
            
        if enable_hybrid:
            self.hybrid_strategy = AdaptiveHybridStrategy()
        else:
            self.hybrid_strategy = None
    
    async def execute(
        self,
        context: RetrievalContext,
        **kwargs
    ) -> RetrievalContext:
        """Execute all enabled enhancements in sequence"""
        # 1. Check semantic cache
        if self.cache_strategy:
            context = await self.cache_strategy.execute(context, **kwargs)
            if context.metadata.get("skip_retrieval"):
                return context
        
        # 2. Apply query expansion
        if self.expansion_strategy:
            context = await self.expansion_strategy.execute(context, **kwargs)
        
        # 3. Apply adaptive hybrid fusion (if results available)
        if self.hybrid_strategy and context.metadata.get("dense_results"):
            context = await self.hybrid_strategy.execute(context, **kwargs)
        
        # 4. Apply contextual scoring
        if self.contextual_strategy and context.documents:
            context = await self.contextual_strategy.execute(context, **kwargs)
        
        return context
    
    async def post_process(self, context: RetrievalContext, **kwargs) -> None:
        """Post-process for all strategies"""
        # Cache results
        if self.cache_strategy:
            await self.cache_strategy.post_process(context, **kwargs)
        
        # Update conversation memory
        if self.contextual_strategy:
            await self.contextual_strategy.post_process(context, **kwargs)