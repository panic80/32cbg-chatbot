"""Unified retriever implementation that uses a strategy pipeline pattern."""

from collections import deque, OrderedDict
from typing import List, Dict, Any, Optional, Type
import asyncio
from datetime import datetime

# Maximum number of timing samples to keep per strategy
MAX_TIMING_HISTORY = 100

from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import Field

from app.components.base import BaseRAGRetriever, BaseComponent
from app.core.logging import get_logger
from app.unified_retrieval.strategies.base import (
    RetrievalContext,
    StrategyPipeline,
    PipelineConfig,
    BaseStrategy,
    StrategyType
)

logger = get_logger(__name__)


class LRUDocumentCache:
    """LRU cache for document query results with TTL support."""

    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self._cache: OrderedDict[str, tuple[List[Document], float]] = OrderedDict()

    def get(self, key: str) -> Optional[List[Document]]:
        """Get documents from cache if present and not expired."""
        if key not in self._cache:
            return None
        docs, timestamp = self._cache[key]
        # Check TTL
        if datetime.utcnow().timestamp() - timestamp > self.ttl:
            del self._cache[key]
            return None
        # Move to end (most recently used)
        self._cache.move_to_end(key)
        return docs

    def set(self, key: str, docs: List[Document]) -> None:
        """Store documents in cache, evicting oldest if at capacity."""
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = (docs, datetime.utcnow().timestamp())
        # Evict oldest entries if over capacity
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)


class UnifiedRetriever(BaseRAGRetriever):
    """
    Unified retriever that uses a configurable strategy pipeline.
    
    This retriever can be configured to use different combinations of strategies
    for query enhancement, filtering, retrieval, scoring, and post-processing.
    It maintains compatibility with the existing parallel pipeline while offering
    a more flexible and extensible architecture.
    """
    
    # Configuration
    name: str = Field(default="UnifiedRetriever", description="Retriever name")
    description: str = Field(
        default="Unified strategy-based retriever",
        description="Retriever description"
    )
    
    # Pipeline configuration
    pipeline_config: Optional[PipelineConfig] = Field(
        default=None,
        description="Pipeline configuration"
    )
    pipeline: Optional[StrategyPipeline] = Field(
        default=None,
        description="Strategy pipeline instance"
    )
    
    # Fallback configuration
    fallback_retriever: Optional[BaseRAGRetriever] = Field(
        default=None,
        description="Fallback retriever if pipeline fails"
    )
    
    # Resource dependencies (set after initialization)
    vectorstore: Optional[Any] = Field(default=None, exclude=True)
    llm: Optional[Any] = Field(default=None, exclude=True)
    embeddings: Optional[Any] = Field(default=None, exclude=True)
    
    # Performance settings
    enable_caching: bool = Field(
        default=True,
        description="Enable query result caching"
    )
    cache_ttl: int = Field(
        default=3600,
        description="Cache TTL in seconds"
    )
    
    # Monitoring
    _pipeline_metrics: Dict[str, Any] = {}
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
        
    def __init__(self, **data):
        """Initialize the unified retriever."""
        super().__init__(**data)
        
        # Pipeline will be built later after resources are set
        self._resources_set = False
            
        # Initialize metrics
        self._pipeline_metrics = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "strategy_timings": {},
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # Initialize cache if enabled
        if self.enable_caching:
            self._init_cache()
            
    def _init_cache(self) -> None:
        """Initialize the LRU query cache."""
        self._query_cache = LRUDocumentCache(max_size=1000, ttl=self.cache_ttl)
        
    def _get_cache_key(self, query: str, **kwargs) -> str:
        """Generate cache key for a query."""
        import hashlib
        import json
        
        # Include relevant kwargs in cache key
        cache_data = {
            "query": query,
            "top_k": kwargs.get("top_k", 10),
            "filters": kwargs.get("filters", {})
        }
        
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()
        
    def _check_cache(self, cache_key: str) -> Optional[List[Document]]:
        """Check if query result is in LRU cache."""
        if not self.enable_caching:
            return None

        docs = self._query_cache.get(cache_key)
        if docs is not None:
            self._pipeline_metrics["cache_hits"] += 1
            logger.debug(f"Cache hit for key: {cache_key}")
            return docs

        self._pipeline_metrics["cache_misses"] += 1
        return None
        
    def _update_cache(self, cache_key: str, documents: List[Document]) -> None:
        """Update LRU cache with query results."""
        if not self.enable_caching:
            return
        self._query_cache.set(cache_key, documents)
                
    def add_strategy(self, strategy: BaseStrategy) -> None:
        """Add a strategy to the pipeline."""
        if not self.pipeline:
            self.pipeline = StrategyPipeline(name=f"{self.name}_pipeline")
        self.pipeline.add_strategy(strategy)
        
    def set_pipeline(self, pipeline: StrategyPipeline) -> None:
        """Set the strategy pipeline."""
        self.pipeline = pipeline
        
    def set_pipeline_config(self, config: PipelineConfig) -> None:
        """Set pipeline configuration and build pipeline."""
        self.pipeline_config = config
        self.pipeline = config.build_pipeline()
        
    def _ensure_pipeline_built(self) -> None:
        """Ensure the pipeline is built with resources available."""
        if self.pipeline_config and not self.pipeline:
            self.pipeline = self.pipeline_config.build_pipeline()
            
        # Pass resources to strategies that need them
        if self.pipeline:
            for strategy in self.pipeline.get_all_strategies():
                if hasattr(strategy, 'llm') and strategy.llm is None and hasattr(self, 'llm'):
                    strategy.llm = self.llm
                if hasattr(strategy, 'vectorstore') and strategy.vectorstore is None and hasattr(self, 'vectorstore'):
                    strategy.vectorstore = self.vectorstore
                if hasattr(strategy, 'embeddings') and strategy.embeddings is None and hasattr(self, 'embeddings'):
                    strategy.embeddings = self.embeddings
        
    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
        **kwargs: Any
    ) -> List[Document]:
        """
        Get relevant documents using the strategy pipeline.
        
        Args:
            query: The search query
            run_manager: Callback manager
            **kwargs: Additional arguments
            
        Returns:
            List of relevant documents
        """
        # Ensure pipeline is built with resources
        self._ensure_pipeline_built()
        
        start_time = datetime.utcnow()
        self._pipeline_metrics["total_queries"] += 1
        
        try:
            # Check cache first
            cache_key = self._get_cache_key(query, **kwargs)
            cached_docs = self._check_cache(cache_key)
            if cached_docs is not None:
                return cached_docs
                
            # Ensure we have a pipeline
            if not self.pipeline:
                if self.fallback_retriever:
                    logger.warning(
                        f"{self.name}: No pipeline configured, using fallback retriever"
                    )
                    # Remove run_manager from kwargs for fallback retriever
                    fallback_kwargs = {k: v for k, v in kwargs.items() if k != "run_manager"}
                    return await self.fallback_retriever.aget_relevant_documents(
                        query,
                        run_manager=run_manager,
                        **fallback_kwargs
                    )
                else:
                    raise ValueError(
                        f"{self.name}: No pipeline or fallback retriever configured"
                    )
                    
            # Create retrieval context
            # Remove run_manager from kwargs to avoid passing it twice
            search_kwargs = {k: v for k, v in kwargs.items() if k != "run_manager"}
            
            context = RetrievalContext(
                original_query=query,
                top_k=kwargs.get("top_k", 10),
                search_kwargs=search_kwargs,
                metadata={
                    "retriever_name": self.name,
                    "run_manager": run_manager,
                    "start_time": start_time.isoformat()
                }
            )
            
            # Add any filters from kwargs
            if "filters" in kwargs:
                context.filters.update(kwargs["filters"])
                
            # Execute pipeline
            logger.info(f"{self.name}: Executing pipeline for query: {query[:100]}...")
            result_context = await self.pipeline.execute(context)
            
            # Get final documents
            documents = result_context.documents[:context.top_k]
            
            # Update cache
            self._update_cache(cache_key, documents)
            
            # Update metrics
            self._pipeline_metrics["successful_queries"] += 1
            elapsed = (datetime.utcnow() - start_time).total_seconds()
            
            # Track strategy timings from context metrics (bounded to prevent memory leak)
            if "strategy_timings" in result_context.metrics:
                for strategy, timing in result_context.metrics["strategy_timings"].items():
                    if strategy not in self._pipeline_metrics["strategy_timings"]:
                        self._pipeline_metrics["strategy_timings"][strategy] = deque(maxlen=MAX_TIMING_HISTORY)
                    self._pipeline_metrics["strategy_timings"][strategy].append(timing)
                    
            logger.info(
                f"{self.name}: Retrieved {len(documents)} documents in {elapsed:.2f}s "
                f"(pipeline errors: {len(result_context.errors)})"
            )
            
            # Log any pipeline errors as warnings
            for error in result_context.errors:
                logger.warning(
                    f"{self.name}: Pipeline error in {error['strategy']}: {error['error']}"
                )
                
            return documents
            
        except Exception as e:
            self._pipeline_metrics["failed_queries"] += 1
            logger.error(f"{self.name}: Pipeline execution failed: {e}")
            
            # Try fallback retriever if available
            if self.fallback_retriever:
                logger.info(f"{self.name}: Using fallback retriever")
                # Remove run_manager from kwargs for fallback retriever
                fallback_kwargs = {k: v for k, v in kwargs.items() if k != "run_manager"}
                return await self.fallback_retriever.aget_relevant_documents(
                    query,
                    run_manager=run_manager,
                    **fallback_kwargs
                )
            else:
                raise
                
    def get_pipeline_metrics(self) -> Dict[str, Any]:
        """Get detailed pipeline metrics."""
        metrics = self._pipeline_metrics.copy()
        
        # Calculate average strategy timings
        if metrics["strategy_timings"]:
            avg_timings = {}
            for strategy, timings in metrics["strategy_timings"].items():
                if timings:
                    avg_timings[strategy] = sum(timings) / len(timings)
            metrics["avg_strategy_timings"] = avg_timings
            
        # Add cache efficiency
        total_cache_ops = metrics["cache_hits"] + metrics["cache_misses"]
        if total_cache_ops > 0:
            metrics["cache_hit_rate"] = metrics["cache_hits"] / total_cache_ops
        else:
            metrics["cache_hit_rate"] = 0.0
            
        # Add base retriever metrics
        metrics.update(super().get_metrics())
        
        # Add pipeline component metrics if available
        if self.pipeline:
            metrics["pipeline_metrics"] = self.pipeline.get_metrics()
            
        return metrics
        
    def clear_cache(self) -> None:
        """Clear the query cache."""
        if hasattr(self, "_query_cache"):
            self._query_cache.clear()
            logger.info(f"{self.name}: Query cache cleared")
            
    def reset_metrics(self) -> None:
        """Reset all metrics."""
        super().reset_metrics()
        self._pipeline_metrics = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "strategy_timings": {},
            "cache_hits": 0,
            "cache_misses": 0
        }
        if self.pipeline:
            self.pipeline.reset_metrics()


class UnifiedRetrieverBuilder:
    """Builder class for creating unified retrievers with common configurations."""
    
    @staticmethod
    def from_config(config: Dict[str, Any]) -> UnifiedRetriever:
        """
        Create a unified retriever from a configuration dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Configured UnifiedRetriever instance
        """
        # Extract pipeline config if present
        pipeline_config = None
        if "pipeline" in config:
            pipeline_config = PipelineConfig(**config["pipeline"])
            
        # Create retriever
        retriever_config = {
            k: v for k, v in config.items()
            if k not in ["pipeline", "fallback_retriever_config"]
        }
        
        retriever = UnifiedRetriever(
            pipeline_config=pipeline_config,
            **retriever_config
        )
        
        # Add fallback retriever if configured
        if "fallback_retriever_config" in config:
            fallback_config = config["fallback_retriever_config"]
            fallback_class = fallback_config.get("class")
            if fallback_class:
                # Import and instantiate fallback retriever
                from importlib import import_module
                module_name, class_name = fallback_class.rsplit(".", 1)
                module = import_module(module_name)
                retriever_class = getattr(module, class_name)
                
                fallback_params = {
                    k: v for k, v in fallback_config.items()
                    if k != "class"
                }
                retriever.fallback_retriever = retriever_class(**fallback_params)
                
        return retriever
        
    @staticmethod
    def create_simple_retriever(
        name: str,
        retrieval_strategy: BaseStrategy,
        query_enhancement_strategy: Optional[BaseStrategy] = None,
        scoring_strategy: Optional[BaseStrategy] = None,
        **kwargs
    ) -> UnifiedRetriever:
        """
        Create a simple unified retriever with basic strategies.
        
        Args:
            name: Retriever name
            retrieval_strategy: Core retrieval strategy
            query_enhancement_strategy: Optional query enhancement
            scoring_strategy: Optional scoring/reranking
            **kwargs: Additional retriever arguments
            
        Returns:
            Configured UnifiedRetriever instance
        """
        # Create pipeline
        strategies = []
        
        if query_enhancement_strategy:
            strategies.append(query_enhancement_strategy)
            
        strategies.append(retrieval_strategy)
        
        if scoring_strategy:
            strategies.append(scoring_strategy)
            
        pipeline = StrategyPipeline(
            strategies=strategies,
            name=f"{name}_pipeline"
        )
        
        # Create retriever
        return UnifiedRetriever(
            name=name,
            pipeline=pipeline,
            **kwargs
        )
        
    @staticmethod
    def create_parallel_retriever(
        name: str,
        retrieval_strategies: List[BaseStrategy],
        merge_strategy: Optional[BaseStrategy] = None,
        **kwargs
    ) -> UnifiedRetriever:
        """
        Create a unified retriever that executes multiple retrieval strategies in parallel.
        
        Args:
            name: Retriever name
            retrieval_strategies: List of retrieval strategies to run in parallel
            merge_strategy: Optional strategy to merge/rerank results
            **kwargs: Additional retriever arguments
            
        Returns:
            Configured UnifiedRetriever instance
        """
        # Create pipeline with parallel group
        pipeline = StrategyPipeline(
            strategies=[merge_strategy] if merge_strategy else [],
            name=f"{name}_pipeline",
            parallel_groups={
                "parallel_retrieval": retrieval_strategies
            }
        )
        
        # Create retriever
        return UnifiedRetriever(
            name=name,
            pipeline=pipeline,
            **kwargs
        )