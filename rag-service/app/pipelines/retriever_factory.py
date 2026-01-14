"""
Hybrid Retriever Factory for creating configurable retriever chains.

This factory provides a flexible way to create different retriever configurations
based on requirements and available resources.
"""

import time
from typing import Dict, Any, Optional, List, Union
from enum import Enum
import logging

from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models import BaseLLM
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.retrievers import BM25Retriever

from app.components.ensemble_retriever import ContentBoostedEnsembleRetriever
from app.components.multi_query_retriever import MultiQueryRetriever
from app.components.contextual_compressor import TravelContextualCompressor
from app.components.self_query_retriever import TravelSelfQueryRetriever
from app.components.parent_document_retriever import TravelParentDocumentRetriever
from app.components.bm25_retriever import TravelBM25Retriever
from app.components.cooccurrence_retriever import TravelCooccurrenceRetriever
from app.components.authority_reranker import AuthorityReranker, AuthorityRerankingRetriever
from app.core.logging import get_logger
from app.core.config import settings

# Import unified retrieval components
from app.unified_retrieval.unified_retriever import UnifiedRetriever, UnifiedRetrieverBuilder
from app.unified_retrieval.strategies.base import PipelineConfig, StrategyConfig, StrategyType

logger = get_logger(__name__)


class RetrieverMode(Enum):
    """Available retriever modes."""
    SIMPLE = "simple"  # Just vector search
    HYBRID = "hybrid"  # Vector + BM25
    ADVANCED = "advanced"  # Full multi-stage pipeline
    CUSTOM = "custom"  # Custom configuration
    UNIFIED = "unified"  # New unified retrieval system


class RetrieverConfig:
    """Configuration for retriever creation."""
    
    def __init__(
        self,
        mode: RetrieverMode = RetrieverMode.ADVANCED,
        use_bm25: bool = True,
        use_multi_query: bool = False,  # Disabled for performance (saves ~6s)
        use_self_query: bool = True,
        use_compression: bool = True,
        use_smart_chunking: bool = True,
        use_cooccurrence: bool = True,
        use_reranking: bool = True,
        compression_mode: str = "hybrid",
        ensemble_weights: Optional[Dict[str, float]] = None,
        k: int = 10,
        enable_profiling: bool = True,
        unified_config: Optional[Dict[str, Any]] = None
    ):
        self.mode = mode
        self.use_bm25 = use_bm25
        self.use_multi_query = use_multi_query
        self.use_self_query = use_self_query
        self.use_compression = use_compression
        self.use_smart_chunking = use_smart_chunking
        self.use_cooccurrence = use_cooccurrence
        self.use_reranking = use_reranking
        self.compression_mode = compression_mode
        self.ensemble_weights = ensemble_weights or {
            "vector": 0.35,
            "bm25": 0.35,
            "mmr": 0.20,
            "smart_chunk": 0.10
        }
        self.k = k
        self.enable_profiling = enable_profiling
        self.unified_config = unified_config or {}


class HybridRetrieverFactory:
    """Factory for creating configurable retriever chains."""
    
    def __init__(
        self,
        vectorstore: VectorStore,
        llm: Optional[BaseLLM] = None,
        embeddings: Optional[Embeddings] = None
    ):
        """
        Initialize the retriever factory.
        
        Args:
            vectorstore: The vector store to use
            llm: Language model for advanced retrievers
            embeddings: Embeddings model for compression
        """
        self.vectorstore = vectorstore
        self.llm = llm
        self.embeddings = embeddings
        self._profiling_data = {}
    
    def create_retriever(
        self,
        config: Optional[Union[RetrieverConfig, Dict[str, Any]]] = None
    ) -> BaseRetriever:
        """
        Create a retriever based on the configuration.
        
        Args:
            config: Retriever configuration (uses defaults if not provided)
                   Can be a RetrieverConfig object or a dict
            
        Returns:
            Configured retriever chain
        """
        # Handle dict configuration
        if isinstance(config, dict):
            return self._create_retriever_from_dict(config)
        
        config = config or RetrieverConfig()
        
        if config.enable_profiling:
            start_time = time.time()
        
        # Create retriever based on mode
        if config.mode == RetrieverMode.SIMPLE:
            retriever = self._create_simple_retriever(config)
        elif config.mode == RetrieverMode.HYBRID:
            retriever = self._create_hybrid_retriever(config)
        elif config.mode == RetrieverMode.ADVANCED:
            retriever = self._create_advanced_retriever(config)
        elif config.mode == RetrieverMode.UNIFIED:
            retriever = self._create_unified_retriever(config)
        else:  # CUSTOM
            retriever = self._create_custom_retriever(config)
        
        if config.enable_profiling:
            self._profiling_data["creation_time"] = time.time() - start_time
            logger.info(f"Retriever created in {self._profiling_data['creation_time']:.2f}s")
        
        return retriever
    
    def _create_retriever_from_dict(self, config: Dict[str, Any]) -> BaseRetriever:
        """Create a retriever from a dictionary configuration."""
        retriever_type = config.get("type", "vector")
        k = config.get("k", 10)
        
        if retriever_type == "vector":
            search_type = config.get("search_type", "similarity")
            search_kwargs = {"k": k}
            if search_type == "mmr":
                search_kwargs["lambda_mult"] = config.get("lambda_mult", 0.5)
            return self.vectorstore.as_retriever(
                search_type=search_type,
                search_kwargs=search_kwargs
            )
        
        elif retriever_type == "bm25":
            if not settings.enable_bm25:
                logger.info("BM25 disabled by configuration; using vector retriever")
                return self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": k}
                )
            try:
                return TravelBM25Retriever(k=k)
            except Exception as e:
                logger.warning(f"Failed to create BM25 retriever (likely no index): {e}")
                # Fallback to vector retriever
                return self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": k}
                )
        
        elif retriever_type == "multi_query":
            base_retriever_config = config.get("base_retriever", "vector_similarity")
            if isinstance(base_retriever_config, str):
                # Create base retriever from string reference
                base_config = {"type": "vector", "search_type": "similarity", "k": k}
                base_retriever = self._create_retriever_from_dict(base_config)
            else:
                base_retriever = self._create_retriever_from_dict(base_retriever_config)
            
            llm = config.get("llm", self.llm)
            if not llm:
                logger.warning("Multi-query retriever requested but no LLM provided")
                return base_retriever
                
            return MultiQueryRetriever(
                retriever=base_retriever,
                llm=llm,
                include_original=True
            )
        
        else:
            logger.warning(f"Unknown retriever type: {retriever_type}, using vector similarity")
            return self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k}
            )
    
    def _create_simple_retriever(self, config: RetrieverConfig) -> BaseRetriever:
        """Create a simple vector search retriever."""
        logger.info("Creating simple vector retriever")
        
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": config.k}
        )
        
        # Add compression if requested
        if config.use_compression and self.llm and self.embeddings:
            retriever = self._add_compression(retriever, config)
        
        return retriever
    
    def _create_hybrid_retriever(self, config: RetrieverConfig) -> BaseRetriever:
        """Create a hybrid retriever with vector + BM25."""
        logger.info("Creating hybrid retriever")
        
        retrievers = []
        weights = []
        
        # Vector retriever
        vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": config.k}
        )
        retrievers.append(vector_retriever)
        weights.append(config.ensemble_weights.get("vector", 0.5))
        
        # BM25 retriever
        if config.use_bm25 and settings.enable_bm25:
            try:
                bm25_retriever = TravelBM25Retriever(k=config.k)
                retrievers.append(bm25_retriever)
                weights.append(config.ensemble_weights.get("bm25", 0.5))
            except Exception as e:
                logger.warning(f"Failed to create BM25 retriever: {e}")
        elif config.use_bm25 and not settings.enable_bm25:
            logger.info("BM25 disabled by configuration; skipping BM25 retriever")
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Create ensemble
        ensemble = ContentBoostedEnsembleRetriever(
            retrievers=retrievers,
            weights=weights,
            k=config.k
        )
        
        # Add compression if requested
        if config.use_compression and self.llm and self.embeddings:
            ensemble = self._add_compression(ensemble, config)
        
        return ensemble
    
    def _create_advanced_retriever(self, config: RetrieverConfig) -> BaseRetriever:
        """Create an advanced multi-stage retriever pipeline."""
        logger.info("Creating advanced retriever pipeline")
        
        # Stage 1: Create base retrievers
        base_retrievers = self._create_base_retrievers(config)
        
        # Stage 2: Create ensemble
        ensemble = self._create_ensemble(base_retrievers, config)
        
        # Stage 3: Add advanced features
        retriever = ensemble
        
        # Add co-occurrence retrieval
        if config.use_cooccurrence:
            try:
                # Co-occurrence retriever needs documents, not a base retriever
                # We'll add it to the ensemble instead of wrapping
                # Note: CooccurrenceRetriever will need similar updates to load from disk/artifacts
                # For now, we'll skip if it requires explicit documents
                pass
            except Exception as e:
                logger.warning(f"Failed to add co-occurrence retriever: {e}")
        
        # Add multi-query
        if config.use_multi_query and self.llm:
            try:
                multi_query = MultiQueryRetriever(
                    retriever=retriever,  # Correct parameter name
                    llm=self.llm,
                    include_original=True
                )
                retriever = multi_query
            except Exception as e:
                logger.warning(f"Failed to add multi-query retriever: {e}")
        
        # Add compression
        if config.use_compression and self.llm and self.embeddings:
            retriever = self._add_compression(retriever, config)
        
        # Add reranking
        if config.use_reranking:
            try:
                retriever = AuthorityRerankingRetriever(
                    base_retriever=retriever,
                    boost_factor=2.0
                )
            except Exception as e:
                logger.warning(f"Failed to add reranking: {e}")
        
        return retriever
    
    def _create_custom_retriever(self, config: RetrieverConfig) -> BaseRetriever:
        """Create a custom retriever based on specific configuration."""
        logger.info("Creating custom retriever")
        
        # Start with base retrievers
        base_retrievers = self._create_base_retrievers(config)
        
        # Create ensemble if multiple retrievers
        if len(base_retrievers) > 1:
            retriever = self._create_ensemble(base_retrievers, config)
        else:
            retriever = list(base_retrievers.values())[0]
        
        # Add components based on configuration
        if config.use_multi_query and self.llm:
            retriever = MultiQueryRetriever(
                retriever=retriever,  # Correct parameter name
                llm=self.llm,
                include_original=True
            )
        
        if config.use_compression and self.llm and self.embeddings:
            retriever = self._add_compression(retriever, config)
        
        return retriever
    
    def _create_base_retrievers(self, config: RetrieverConfig) -> Dict[str, BaseRetriever]:
        """Create base retrievers for ensemble."""
        retrievers = {}
        
        # Vector retriever
        retrievers["vector"] = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": config.k}
        )
        
        # MMR retriever
        retrievers["mmr"] = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": config.k, "lambda_mult": 0.5}
        )
        
        # BM25 retriever
        if config.use_bm25 and settings.enable_bm25:
            try:
                retrievers["bm25"] = TravelBM25Retriever(k=config.k)
            except Exception as e:
                logger.warning(f"Failed to create BM25 retriever: {e}")
        elif config.use_bm25 and not settings.enable_bm25:
            logger.info("BM25 disabled by configuration; skipping BM25 retriever")
        
        # Smart chunk retriever (parent document retriever)
        if config.use_smart_chunking:
            try:
                # Create child retriever for finding chunks
                child_retriever = self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": config.k // 2}
                )
                
                retrievers["smart_chunk"] = TravelParentDocumentRetriever(
                    child_retriever=child_retriever,
                    k=config.k
                )
            except Exception as e:
                logger.warning(f"Failed to create smart chunk retriever: {e}")
        
        # Self-query retriever
        if config.use_self_query and self.llm:
            try:
                retrievers["self_query"] = TravelSelfQueryRetriever(
                    vectorstore=self.vectorstore,
                    llm=self.llm,
                    document_contents="Canadian Forces travel instructions",
                    search_kwargs={"k": config.k}
                )
            except Exception as e:
                logger.warning(f"Failed to create self-query retriever: {e}")
        
        return retrievers
    
    def _create_ensemble(
        self,
        base_retrievers: Dict[str, BaseRetriever],
        config: RetrieverConfig
    ) -> BaseRetriever:
        """Create ensemble from base retrievers."""
        retrievers = []
        weights = []
        
        for name, retriever in base_retrievers.items():
            retrievers.append(retriever)
            weights.append(config.ensemble_weights.get(name, 0.2))
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        return ContentBoostedEnsembleRetriever(
            retrievers=retrievers,
            weights=weights,
            k=config.k
        )
    
    def _add_compression(
        self,
        retriever: BaseRetriever,
        config: RetrieverConfig
    ) -> BaseRetriever:
        """Add compression to a retriever."""
        try:
            compressor = TravelContextualCompressor(
                base_retriever=retriever,
                llm=self.llm,
                embeddings=self.embeddings
            )
            
            return compressor
        except Exception as e:
            logger.warning(f"Failed to add compression: {e}")
            return retriever
    
    def _create_unified_retriever(self, config: RetrieverConfig) -> BaseRetriever:
        """Create a unified retriever using the new strategy-based system."""
        logger.info("Creating unified retriever")
        
        # Check if we have a pipeline config in unified_config
        if "pipeline_config" in config.unified_config:
            # Use the provided pipeline configuration
            pipeline_config = PipelineConfig(**config.unified_config["pipeline_config"])
            retriever = UnifiedRetriever(
                name=config.unified_config.get("name", "unified_retriever"),
                pipeline_config=pipeline_config,
                enable_caching=config.unified_config.get("enable_caching", True),
                cache_ttl=config.unified_config.get("cache_ttl", 3600)
            )
            # Store component references for strategies to use
            retriever.vectorstore = self.vectorstore
            retriever.llm = self.llm
            retriever.embeddings = self.embeddings
        else:
            # Map existing configuration to unified strategies
            retriever = self._create_unified_from_legacy_config(config)
            
        # Add fallback retriever if requested
        if config.unified_config.get("use_fallback", True):
            # Create a simple retriever as fallback
            fallback_config = RetrieverConfig(
                mode=RetrieverMode.SIMPLE,
                k=config.k,
                enable_profiling=False
            )
            fallback_retriever = self._create_simple_retriever(fallback_config)
            retriever.fallback_retriever = fallback_retriever
            
        return retriever
    
    def _create_unified_from_legacy_config(self, config: RetrieverConfig) -> UnifiedRetriever:
        """Create a unified retriever by mapping legacy configuration."""
        from app.unified_retrieval.migration import LegacyConfigMapper
        
        # Create mapper
        mapper = LegacyConfigMapper(
            vectorstore=self.vectorstore,
            llm=self.llm,
            embeddings=self.embeddings
        )
        
        # Map configuration
        unified_retriever = mapper.map_retriever_config(config)
        
        return unified_retriever
    
    def get_profiling_data(self) -> Dict[str, Any]:
        """Get profiling data from the last retriever creation."""
        return self._profiling_data
    
    @classmethod
    def create_from_vector_store_manager(
        cls,
        vector_store_manager,
        llm: Optional[BaseLLM] = None
    ) -> "HybridRetrieverFactory":
        """
        Convenience method to create factory from VectorStoreManager.
        
        Args:
            vector_store_manager: VectorStoreManager instance
            llm: Language model
            
        Returns:
            HybridRetrieverFactory instance
        """
        return cls(
            vectorstore=vector_store_manager.vector_store,
            llm=llm,
            embeddings=vector_store_manager.embeddings
        )


# Example usage functions
def create_simple_retriever(vectorstore: VectorStore) -> BaseRetriever:
    """Create a simple vector search retriever."""
    factory = HybridRetrieverFactory(vectorstore)
    config = RetrieverConfig(mode=RetrieverMode.SIMPLE)
    return factory.create_retriever(config)
