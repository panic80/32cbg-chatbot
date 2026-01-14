"""Base strategy classes for the unified retrieval framework."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Type, Callable
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from datetime import datetime

from langchain_core.documents import Document
from pydantic import BaseModel, Field

from app.components.base import BaseComponent
from app.core.logging import get_logger

logger = get_logger(__name__)


class StrategyType(Enum):
    """Types of strategies in the retrieval pipeline."""
    QUERY_ENHANCEMENT = "query_enhancement"
    FILTERING = "filtering"
    RETRIEVAL = "retrieval"
    SCORING = "scoring"
    RERANKING = "reranking"
    POST_PROCESSING = "post_processing"


@dataclass
class RetrievalContext:
    """
    Context object that gets passed between strategies in the pipeline.
    
    This contains all the data and metadata needed for retrieval operations,
    allowing strategies to share information and build upon each other's work.
    """
    # Core retrieval data
    original_query: str
    enhanced_query: Optional[str] = None
    documents: List[Document] = field(default_factory=list)
    
    # Metadata and configuration
    metadata: Dict[str, Any] = field(default_factory=dict)
    filters: Dict[str, Any] = field(default_factory=dict)
    scores: Dict[str, float] = field(default_factory=dict)
    
    # Pipeline control
    strategy_outputs: Dict[str, Any] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    # User parameters
    top_k: int = 10
    search_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    def add_strategy_output(self, strategy_name: str, output: Any) -> None:
        """Add output from a strategy execution."""
        self.strategy_outputs[strategy_name] = output
        
    def add_error(self, strategy_name: str, error: Exception) -> None:
        """Record an error from a strategy."""
        self.errors.append({
            "strategy": strategy_name,
            "error": str(error),
            "timestamp": datetime.utcnow().isoformat()
        })
        
    def add_metric(self, key: str, value: Any) -> None:
        """Add a metric to the context."""
        self.metrics[key] = value
        
    def get_query(self) -> str:
        """Get the current query (enhanced if available, otherwise original)."""
        return self.enhanced_query or self.original_query


class BaseStrategy(BaseComponent, ABC):
    """
    Base class for all retrieval strategies.
    
    Strategies are composable units that perform specific operations
    in the retrieval pipeline.
    """
    
    def __init__(
        self,
        strategy_type: StrategyType,
        name: Optional[str] = None,
        required_inputs: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize the strategy.
        
        Args:
            strategy_type: Type of strategy
            name: Name of the strategy instance
            required_inputs: List of required keys in context
            **kwargs: Additional configuration
        """
        name = name or f"{strategy_type.value}_strategy"
        super().__init__(
            component_type="strategy",
            component_name=name,
            **kwargs
        )
        self.strategy_type = strategy_type
        self.required_inputs = required_inputs or []
        self.config = kwargs
        
    def validate_context(self, context: RetrievalContext) -> None:
        """
        Validate that the context has all required inputs.
        
        Args:
            context: The retrieval context
            
        Raises:
            ValueError: If required inputs are missing
        """
        for required in self.required_inputs:
            if required == "documents" and not context.documents:
                raise ValueError(f"Strategy {self.component_name} requires documents in context")
            elif required == "enhanced_query" and not context.enhanced_query:
                raise ValueError(f"Strategy {self.component_name} requires enhanced_query in context")
            elif required not in context.metadata:
                raise ValueError(f"Strategy {self.component_name} requires {required} in context metadata")
    
    @abstractmethod
    async def execute(self, context: RetrievalContext) -> RetrievalContext:
        """
        Execute the strategy on the given context.
        
        Args:
            context: The retrieval context
            
        Returns:
            Updated retrieval context
        """
        pass
        
    @BaseComponent.monitor_performance
    async def __call__(self, context: RetrievalContext) -> RetrievalContext:
        """
        Execute the strategy with validation and error handling.
        
        Args:
            context: The retrieval context
            
        Returns:
            Updated retrieval context
        """
        try:
            # Validate inputs
            self.validate_context(context)
            
            # Log execution start
            self._log_event(
                "strategy_start",
                {
                    "strategy_type": self.strategy_type.value,
                    "query": context.get_query()[:100],
                    "doc_count": len(context.documents)
                }
            )
            
            # Execute strategy
            result = await self.execute(context)
            
            # Log execution complete
            self._log_event(
                "strategy_complete",
                {
                    "strategy_type": self.strategy_type.value,
                    "doc_count": len(result.documents),
                    "errors": len(result.errors)
                }
            )
            
            return result
            
        except Exception as e:
            # Log error
            self._log_event(
                "strategy_error",
                {"error": str(e), "strategy_type": self.strategy_type.value},
                level="error"
            )
            
            # Add error to context
            context.add_error(self.component_name, e)
            
            # Re-raise if critical
            if self.config.get("fail_on_error", True):
                raise
                
            return context


class StrategyPipeline(BaseComponent):
    """
    Orchestrates the execution of multiple strategies in sequence.
    
    Strategies are executed in order, with each receiving the context
    modified by previous strategies.
    """
    
    def __init__(
        self,
        strategies: Optional[List[BaseStrategy]] = None,
        name: str = "strategy_pipeline",
        parallel_groups: Optional[Dict[str, List[BaseStrategy]]] = None,
        **kwargs
    ):
        """
        Initialize the pipeline.
        
        Args:
            strategies: List of strategies to execute in sequence
            name: Name of the pipeline
            parallel_groups: Groups of strategies to execute in parallel
            **kwargs: Additional configuration
        """
        super().__init__(
            component_type="pipeline",
            component_name=name,
            **kwargs
        )
        self.strategies = strategies or []
        self.parallel_groups = parallel_groups or {}
        self.config = kwargs
        
    def add_strategy(self, strategy: BaseStrategy) -> None:
        """Add a strategy to the pipeline."""
        self.strategies.append(strategy)
        self._log_event(
            "strategy_added",
            {
                "strategy_name": strategy.component_name,
                "strategy_type": strategy.strategy_type.value,
                "total_strategies": len(self.strategies)
            }
        )
        
    def add_parallel_group(self, group_name: str, strategies: List[BaseStrategy]) -> None:
        """Add a group of strategies to execute in parallel."""
        self.parallel_groups[group_name] = strategies
        self._log_event(
            "parallel_group_added",
            {
                "group_name": group_name,
                "strategy_count": len(strategies)
            }
        )
        
    async def _execute_parallel_group(
        self,
        group_name: str,
        strategies: List[BaseStrategy],
        context: RetrievalContext
    ) -> RetrievalContext:
        """Execute a group of strategies in parallel."""
        self._log_event(
            "parallel_execution_start",
            {
                "group_name": group_name,
                "strategy_count": len(strategies)
            }
        )
        
        # Create context copies for each strategy
        contexts = [RetrievalContext(
            original_query=context.original_query,
            enhanced_query=context.enhanced_query,
            documents=context.documents.copy(),
            metadata=context.metadata.copy(),
            filters=context.filters.copy(),
            scores=context.scores.copy(),
            strategy_outputs=context.strategy_outputs.copy(),
            errors=context.errors.copy(),
            metrics=context.metrics.copy(),
            top_k=context.top_k,
            search_kwargs=context.search_kwargs.copy()
        ) for _ in strategies]
        
        # Execute strategies in parallel
        tasks = [
            strategy(ctx) 
            for strategy, ctx in zip(strategies, contexts)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Merge results back into original context
        merged_docs = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                context.add_error(
                    f"{group_name}_{strategies[i].component_name}",
                    result
                )
            else:
                # Merge documents
                merged_docs.extend(result.documents)
                
                # Merge metadata
                context.metadata.update(result.metadata)
                
                # Merge strategy outputs
                context.strategy_outputs.update(result.strategy_outputs)
                
                # Merge scores
                context.scores.update(result.scores)
                
                # Merge errors
                context.errors.extend(result.errors)
                
        # Deduplicate documents if needed
        if self.config.get("deduplicate_docs", True):
            seen = set()
            unique_docs = []
            for doc in merged_docs:
                doc_id = doc.metadata.get("id", doc.page_content[:100])
                if doc_id not in seen:
                    seen.add(doc_id)
                    unique_docs.append(doc)
            context.documents = unique_docs
        else:
            context.documents = merged_docs
            
        self._log_event(
            "parallel_execution_complete",
            {
                "group_name": group_name,
                "doc_count": len(context.documents),
                "errors": len([r for r in results if isinstance(r, Exception)])
            }
        )
        
        return context
        
    @BaseComponent.monitor_performance
    async def execute(self, context: RetrievalContext) -> RetrievalContext:
        """
        Execute the pipeline on the given context.
        
        Args:
            context: The retrieval context
            
        Returns:
            Updated retrieval context
        """
        self._log_event(
            "pipeline_start",
            {
                "query": context.get_query()[:100],
                "strategy_count": len(self.strategies),
                "parallel_group_count": len(self.parallel_groups)
            }
        )
        
        try:
            # Execute sequential strategies
            for strategy in self.strategies:
                context = await strategy(context)
                
            # Execute parallel groups
            for group_name, group_strategies in self.parallel_groups.items():
                context = await self._execute_parallel_group(
                    group_name,
                    group_strategies,
                    context
                )
                
            self._log_event(
                "pipeline_complete",
                {
                    "doc_count": len(context.documents),
                    "total_errors": len(context.errors),
                    "metrics": context.metrics
                }
            )
            
            return context
            
        except Exception as e:
            self._log_event(
                "pipeline_error",
                {"error": str(e)},
                level="error"
            )
            raise
    
    def get_all_strategies(self) -> List[BaseStrategy]:
        """Get all strategies in the pipeline."""
        all_strategies = list(self.strategies)
        for group in self.parallel_groups.values():
            all_strategies.extend(group)
        return all_strategies


class StrategyConfig(BaseModel):
    """Configuration for a strategy."""
    
    strategy_class: str = Field(description="Fully qualified class name")
    strategy_type: StrategyType = Field(description="Type of strategy")
    enabled: bool = Field(default=True, description="Whether strategy is enabled")
    config: Dict[str, Any] = Field(default_factory=dict, description="Strategy-specific config")
    order: Optional[int] = Field(default=None, description="Execution order (lower first)")
    parallel_group: Optional[str] = Field(default=None, description="Parallel execution group")
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True


class PipelineConfig(BaseModel):
    """Configuration for a strategy pipeline."""
    
    name: str = Field(description="Pipeline name")
    description: str = Field(default="", description="Pipeline description")
    strategies: List[StrategyConfig] = Field(description="List of strategies")
    config: Dict[str, Any] = Field(default_factory=dict, description="Pipeline-specific config")
    
    def build_pipeline(self) -> StrategyPipeline:
        """Build a pipeline from this configuration."""
        # Import strategies dynamically
        from importlib import import_module
        
        # Group strategies
        sequential_strategies = []
        parallel_groups = {}
        
        # Sort by order
        sorted_strategies = sorted(
            [s for s in self.strategies if s.enabled],
            key=lambda x: x.order or 999
        )
        
        for strategy_config in sorted_strategies:
            # Import strategy class
            module_name, class_name = strategy_config.strategy_class.rsplit(".", 1)
            module = import_module(module_name)
            strategy_class = getattr(module, class_name)
            
            # Create strategy instance
            strategy = strategy_class(
                **strategy_config.config
            )
            
            # Add to appropriate group
            if strategy_config.parallel_group:
                if strategy_config.parallel_group not in parallel_groups:
                    parallel_groups[strategy_config.parallel_group] = []
                parallel_groups[strategy_config.parallel_group].append(strategy)
            else:
                sequential_strategies.append(strategy)
                
        # Create pipeline
        pipeline = StrategyPipeline(
            strategies=sequential_strategies,
            name=self.name,
            parallel_groups=parallel_groups,
            **self.config
        )
        
        return pipeline