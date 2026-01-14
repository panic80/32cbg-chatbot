"""Stateful retrieval pipeline with LangGraph orchestration and Redis persistence.

This module wraps ParallelRetrievalPipeline with LangGraph's StateGraph to enable:
- Redis-backed state persistence for conversation continuity
- Iterative refinement cycles when retrieval quality is low
- Automatic query reformulation for better results
"""

import asyncio
import hashlib
import time
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.redis import AsyncRedisSaver

from app.core.config import settings
from app.core.logging import get_logger
from app.pipelines.parallel_retrieval import ParallelRetrievalPipeline
from app.pipelines.query_optimizer import QueryOptimizer
from app.services.performance_monitor import get_performance_monitor
from app.services.cache import get_cache_service

logger = get_logger(__name__)


class RetrievalState(TypedDict):
    """State schema for stateful retrieval workflow."""
    query: str
    original_query: str
    documents: List[Tuple[Document, float]]
    relevance_scores: List[float]
    iteration_count: int
    metadata: Dict[str, Any]
    error: Optional[str]
    finalized: bool


class StatefulRetrievalPipeline:
    """LangGraph-based stateful retrieval pipeline with iterative refinement."""
    
    def __init__(
        self,
        parallel_pipeline: ParallelRetrievalPipeline,
        query_optimizer: Optional[QueryOptimizer] = None,
        redis_url: Optional[str] = None,
        max_iterations: Optional[int] = None,
        relevance_threshold: Optional[float] = None,
        enable_checkpointing: bool = True
    ):
        """Initialize stateful retrieval pipeline.

        Args:
            parallel_pipeline: Underlying parallel retrieval pipeline
            query_optimizer: Query optimizer for refinement
            redis_url: Redis URL for checkpointing (e.g., redis://localhost:6379)
            max_iterations: Maximum refinement iterations (default from settings)
            relevance_threshold: Minimum avg relevance to proceed (default from settings)
            enable_checkpointing: Whether to enable Redis checkpointing
        """
        self.parallel_pipeline = parallel_pipeline
        self.query_optimizer = query_optimizer or QueryOptimizer(llm=None)
        self.max_iterations = max_iterations or settings.max_retrieval_iterations
        self.relevance_threshold = relevance_threshold or settings.relevance_threshold
        self.enable_checkpointing = enable_checkpointing
        self.perf_monitor = get_performance_monitor()

        # Setup checkpointer using official langgraph-checkpoint-redis
        if enable_checkpointing:
            # Try to use existing Redis client from CacheService
            cache_service = get_cache_service()
            if cache_service and cache_service.redis_client:
                try:
                    self.checkpointer = AsyncRedisSaver(cache_service.redis_client)
                    logger.info("Initialized AsyncRedisSaver with existing Redis connection")
                except Exception as e:
                    logger.warning(f"Failed to initialize Redis checkpointer: {e}. Falling back to MemorySaver.")
                    self.checkpointer = MemorySaver()
            else:
                logger.warning("No Redis client available from CacheService, falling back to MemorySaver")
                self.checkpointer = MemorySaver()
        else:
            self.checkpointer = MemorySaver()

        # Build the workflow graph
        self.workflow = self._build_workflow()
        
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow with nodes and edges."""
        workflow = StateGraph(RetrievalState)
        
        # Add nodes
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("assess_quality", self._assess_quality_node)
        workflow.add_node("refine_query", self._refine_query_node)
        workflow.add_node("finalize", self._finalize_node)
        
        # Add edges
        workflow.add_edge("retrieve", "assess_quality")
        
        # Conditional edge from assess_quality
        workflow.add_conditional_edges(
            "assess_quality",
            self._should_refine,
            {
                "refine": "refine_query",
                "finalize": "finalize"
            }
        )
        
        # Loop back from refine to retrieve
        workflow.add_edge("refine_query", "retrieve")
        
        # Terminal edge
        workflow.add_edge("finalize", END)
        
        # Set entry point
        workflow.set_entry_point("retrieve")

        # Compile with checkpointer (official AsyncRedisSaver works directly)
        compiled = workflow.compile(checkpointer=self.checkpointer)

        return compiled
        
    async def _retrieve_node(self, state: RetrievalState, config: Optional[Dict[str, Any]] = None) -> RetrievalState:
        """Retrieve documents using parallel pipeline.
        
        Args:
            state: Current retrieval state
            config: Runtime configuration (contains hyde_generator)
            
        Returns:
            Updated state with retrieved documents
        """
        start_time = time.time()
        query = state["query"]
        iteration = state["iteration_count"]
        
        # Extract hyde_generator from config if available
        hyde_generator = None
        auxiliary_model = None
        auxiliary_provider = None
        if config and "configurable" in config:
            hyde_generator = config["configurable"].get("hyde_generator")
            auxiliary_model = config["configurable"].get("auxiliary_model")
            auxiliary_provider = config["configurable"].get("auxiliary_provider")
        
        logger.info(f"Retrieval iteration {iteration + 1}: query='{query}'")
        
        try:
            # Get HyDE hypothesis from state metadata (only use on first iteration)
            hyde_hypothesis = None
            if iteration == 0:
                hyde_hypothesis = state.get("metadata", {}).get("hyde_hypothesis")

            # Call parallel retrieval pipeline
            documents = await self.parallel_pipeline.retrieve(
                query=query,
                k=settings.retrieval_k,
                merge_strategy="weighted",
                hyde_hypothesis=hyde_hypothesis,
                hyde_generator=hyde_generator,
                auxiliary_model=auxiliary_model,
                auxiliary_provider=auxiliary_provider
            )
            
            state["documents"] = documents
            state["metadata"]["last_retrieval_time"] = time.time() - start_time
            
            # Extract relevance scores from document metadata
            relevance_scores = []
            for doc, score in documents:
                # Try to get relevance score from metadata, fallback to score
                relevance = doc.metadata.get("relevance_score", score)
                relevance_scores.append(float(relevance))
                
            state["relevance_scores"] = relevance_scores
            
            logger.info(f"Retrieved {len(documents)} documents in {state['metadata']['last_retrieval_time']:.2f}s")
            
            # Record metrics
            if self.perf_monitor:
                self.perf_monitor.record_latency(
                    "stateful_retrieval_node_latency_ms",
                    state["metadata"]["last_retrieval_time"] * 1000
                )
                
        except Exception as e:
            logger.error(f"Retrieval failed: {e}", exc_info=True)
            state["error"] = str(e)
            state["documents"] = []
            state["relevance_scores"] = []
            
        return state
        
    async def _assess_quality_node(self, state: RetrievalState) -> RetrievalState:
        """Assess retrieval quality based on relevance scores.
        
        Args:
            state: Current retrieval state
            
        Returns:
            Updated state with quality assessment
        """
        relevance_scores = state.get("relevance_scores", [])
        
        if not relevance_scores:
            state["metadata"]["avg_relevance"] = 0.0
            state["metadata"]["quality_assessment"] = "no_results"
            logger.warning("No relevance scores to assess")
            return state
            
        # Calculate average relevance
        avg_relevance = sum(relevance_scores) / len(relevance_scores)
        state["metadata"]["avg_relevance"] = avg_relevance
        
        # Determine quality
        if avg_relevance >= self.relevance_threshold:
            quality = "acceptable"
        elif state["iteration_count"] >= self.max_iterations - 1:
            quality = "max_iterations_reached"
        else:
            quality = "needs_refinement"
            
        state["metadata"]["quality_assessment"] = quality
        
        logger.info(
            f"Quality assessment: avg_relevance={avg_relevance:.3f}, "
            f"threshold={self.relevance_threshold:.3f}, "
            f"quality={quality}, iteration={state['iteration_count'] + 1}/{self.max_iterations}"
        )
        
        # Record metrics
        if self.perf_monitor:
            self.perf_monitor.record_latency("retrieval_avg_relevance", avg_relevance)
            
        return state
        
    async def _refine_query_node(self, state: RetrievalState) -> RetrievalState:
        """Refine query for better retrieval results.
        
        Args:
            state: Current retrieval state
            
        Returns:
            Updated state with refined query
        """
        iteration = state["iteration_count"]
        original_query = state["original_query"]
        current_query = state["query"]
        
        logger.info(f"Refining query (iteration {iteration + 1})")
        
        try:
            if iteration == 0:
                # First retry: Expand query with synonyms and context
                refined_query = self.query_optimizer.expand_query_for_retry(original_query)
                strategy = "expansion"
            else:
                # Second retry: Simplify to core terms
                refined_query = self.query_optimizer.simplify_query_for_retry(original_query)
                strategy = "simplification"
                
            state["query"] = refined_query
            state["iteration_count"] = iteration + 1
            state["metadata"]["refinement_strategy"] = strategy
            state["metadata"]["previous_queries"] = state["metadata"].get("previous_queries", []) + [current_query]
            
            logger.info(f"Refined query using {strategy}: '{refined_query}'")
            
            # Record metrics
            if self.perf_monitor:
                self.perf_monitor.increment_counter("retrieval_refinements_total")
                
        except Exception as e:
            logger.error(f"Query refinement failed: {e}", exc_info=True)
            # Keep original query on failure
            state["iteration_count"] = iteration + 1
            
        return state
        
    async def _finalize_node(self, state: RetrievalState) -> RetrievalState:
        """Finalize retrieval results.
        
        Args:
            state: Current retrieval state
            
        Returns:
            Finalized state
        """
        state["finalized"] = True
        
        total_iterations = state["iteration_count"] + 1
        avg_relevance = state["metadata"].get("avg_relevance", 0.0)
        
        logger.info(
            f"Finalized retrieval: {len(state['documents'])} documents, "
            f"avg_relevance={avg_relevance:.3f}, iterations={total_iterations}"
        )
        
        # Record final metrics
        if self.perf_monitor:
            self.perf_monitor.record_latency("retrieval_iterations_count", total_iterations)
            if total_iterations > 1:
                self.perf_monitor.increment_counter("retrieval_cycles_triggered_total")
                
        return state
        
    def _should_refine(self, state: RetrievalState) -> str:
        """Decide whether to refine query or finalize.
        
        Args:
            state: Current retrieval state
            
        Returns:
            "refine" to continue refinement, "finalize" to end
        """
        # Check for errors
        if state.get("error"):
            logger.warning(f"Error detected, skipping refinement: {state['error']}")
            return "finalize"
            
        # Check quality assessment
        quality = state["metadata"].get("quality_assessment", "unknown")
        
        if quality == "needs_refinement":
            return "refine"
        else:
            return "finalize"
            
    async def retrieve(
        self,
        query: str,
        k: int = 5,
        session_id: Optional[str] = None,
        merge_strategy: str = "weighted",
        hyde_hypothesis: Optional[str] = None,
        hyde_generator: Optional[Any] = None,
        auxiliary_model: Optional[str] = None,
        auxiliary_provider: Optional[Any] = None
    ) -> List[Tuple[Document, float]]:
        """Execute stateful retrieval with iterative refinement.

        Args:
            query: Search query
            k: Number of documents to return
            session_id: Session ID for persistence (optional)
            merge_strategy: Strategy for merging results (passed to parallel pipeline)
            hyde_hypothesis: Optional HyDE hypothesis for improved retrieval
            hyde_generator: Optional HyDE generator instance for concurrent generation
            auxiliary_model: Optional model for auxiliary tasks
            auxiliary_provider: Optional provider for auxiliary tasks

        Returns:
            List of (document, score) tuples
        """
        start_time = time.time()
        
        # Create thread ID for this retrieval session
        if session_id:
            # Use session + query hash for thread ID
            query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
            thread_id = f"{session_id}:{query_hash}"
        else:
            # Use just query hash
            thread_id = hashlib.md5(query.encode()).hexdigest()
            
        # Pass hyde_generator via configurable config
        config = {
            "configurable": {
                "thread_id": thread_id,
                "hyde_generator": hyde_generator,
                "auxiliary_model": auxiliary_model,
                "auxiliary_provider": auxiliary_provider
            }
        }

        # Initialize state
        initial_state: RetrievalState = {
            "query": query,
            "original_query": query,
            "documents": [],
            "relevance_scores": [],
            "iteration_count": 0,
            "metadata": {
                "start_time": start_time,
                "k": k,
                "merge_strategy": merge_strategy,
                "hyde_hypothesis": hyde_hypothesis
            },
            "error": None,
            "finalized": False
        }

        try:
            # Execute workflow
            logger.info(f"Starting stateful retrieval for query: '{query}' (thread_id={thread_id})")
            
            result = await self.workflow.ainvoke(initial_state, config)

            # Extract documents from final state
            documents = result.get("documents", [])
            
            # Record total latency
            total_latency = time.time() - start_time
            if self.perf_monitor:
                self.perf_monitor.record_latency("stateful_retrieval_total_latency_ms", total_latency * 1000)
                
            logger.info(
                f"Stateful retrieval completed in {total_latency:.2f}s: "
                f"{len(documents)} documents, {result['iteration_count'] + 1} iterations"
            )
            
            return documents[:k]  # Return top-k results
            
        except Exception as e:
            logger.error(f"Stateful retrieval failed: {e}", exc_info=True)
            
            # Fallback to direct parallel pipeline
            logger.info("Falling back to parallel pipeline")
            return await self.parallel_pipeline.retrieve(
                query, 
                k, 
                merge_strategy, 
                hyde_hypothesis=hyde_hypothesis,
                hyde_generator=hyde_generator,
                auxiliary_model=auxiliary_model,
                auxiliary_provider=auxiliary_provider
            )

