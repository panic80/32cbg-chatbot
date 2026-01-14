"""
Gated Parallel Retrieval Coordinator for orchestrating multi-source retrieval.

This component coordinates all retrieval components to execute optimal
retrieval strategies based on query analysis and performance constraints.

Coordination strategy:
- Analyze query with uncertainty scorer and BM25 gate
- Determine optimal K values with adaptive selector
- Execute retrieval in parallel across active retrievers
- Merge results with RRF fusion
- Apply deduplication with conservative thresholds
- Cache results at L2 level for subsequent queries
- Provide detailed execution metrics and reasoning
"""

import asyncio
import time
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from langchain_core.documents import Document

from app.components.uncertainty_scorer import UncertaintyScorer, UncertaintyResult
from app.components.bm25_gating import BM25Gate, BM25GatingResult
from app.components.adaptive_k_selector import AdaptiveKSelector, AdaptiveKResult
from app.components.rrf_merger import RRFMerger, RRFDocument, RRFMergerStats
from app.components.deduplicator import DocumentDeduplicator, DeduplicationStats
from app.services.retrieval_cache import RetrievalL2Cache
from app.core.logging import get_logger
from app.core.config import settings

logger = get_logger(__name__)


@dataclass
class RetrieverExecutionResult:
    """Result from a single retriever execution."""
    retriever_name: str
    documents: List[Document]
    execution_time_ms: float
    documents_requested: int
    documents_returned: int
    success: bool
    error_message: Optional[str] = None
    
    @property
    def retrieval_rate(self) -> float:
        """Documents per second retrieval rate."""
        if self.execution_time_ms <= 0:
            return 0.0
        return (self.documents_returned / self.execution_time_ms) * 1000


@dataclass
class RetrievalCoordinationMetrics:
    """Comprehensive metrics from coordinated retrieval."""
    query: str
    total_execution_time_ms: float
    
    # Analysis phase metrics
    uncertainty_analysis_time_ms: float
    bm25_gating_time_ms: float
    k_selection_time_ms: float
    
    # Execution phase metrics
    cache_lookup_time_ms: float
    retrieval_execution_time_ms: float
    rrf_merge_time_ms: float
    deduplication_time_ms: float
    cache_store_time_ms: float
    
    # Results metrics
    retrievers_executed: List[str]
    total_documents_retrieved: int
    unique_documents_after_dedup: int
    final_documents_returned: int
    
    # Performance metrics
    cache_hit: bool
    deduplication_ratio: float
    average_retrieval_latency_ms: float
    
    # Quality metrics
    uncertainty_score: float
    estimated_recall_coverage: float
    rrf_merger_stats: Optional[RRFMergerStats] = None
    deduplication_stats: Optional[Dict[str, Any]] = None
    
    @property
    def total_speedup_ratio(self) -> float:
        """Ratio of parallel vs sequential execution time."""
        if self.average_retrieval_latency_ms <= 0:
            return 1.0
        sequential_time = len(self.retrievers_executed) * self.average_retrieval_latency_ms
        return sequential_time / max(1.0, self.retrieval_execution_time_ms)


@dataclass
class CoordinatorConfiguration:
    """Configuration for the retrieval coordinator."""
    enable_l2_cache: bool = True
    enable_parallel_execution: bool = True
    max_parallel_workers: int = 4
    retrieval_timeout_ms: float = 5000.0
    enable_detailed_metrics: bool = True
    cache_threshold_docs: int = 5  # Minimum docs to cache
    fallback_on_errors: bool = True
    conservative_deduplication: bool = True
    rrf_score_threshold: float = field(
        default_factory=lambda: getattr(settings, "rrf_score_threshold", 0.15)
    )


class GatedRetrievalCoordinator:
    """
    Coordinates all retrieval components for optimal document retrieval.
    
    This is the main orchestrator that integrates uncertainty scoring,
    BM25 gating, adaptive K-selection, parallel retrieval, RRF merging,
    deduplication, and L2 caching into a cohesive system.
    """
    
    def __init__(
        self,
        # Core analysis components
        uncertainty_scorer: Optional[UncertaintyScorer] = None,
        bm25_gate: Optional[BM25Gate] = None,
        adaptive_k_selector: Optional[AdaptiveKSelector] = None,
        
        # Processing components
        rrf_merger: Optional[RRFMerger] = None,
        deduplicator: Optional[DocumentDeduplicator] = None,
        l2_cache: Optional[RetrievalL2Cache] = None,
        
        # Retriever functions (to be injected)
        dense_retriever: Optional[callable] = None,
        sparse_retriever: Optional[callable] = None,
        bm25_retriever: Optional[callable] = None,
        hybrid_retriever: Optional[callable] = None,
        
        # Configuration
        config: Optional[CoordinatorConfiguration] = None
    ):
        """
        Initialize the gated retrieval coordinator.
        
        Args:
            uncertainty_scorer: Query uncertainty analyzer
            bm25_gate: BM25 activation gate
            adaptive_k_selector: Dynamic K selector
            rrf_merger: Result fusion component
            deduplicator: Document deduplication
            l2_cache: Level 2 retrieval cache
            dense_retriever: Semantic/dense retrieval function
            sparse_retriever: Sparse/lexical retrieval function
            bm25_retriever: BM25 keyword retrieval function
            hybrid_retriever: Hybrid retrieval function
            config: Coordinator configuration
        """
        # Configuration
        self.config = config or CoordinatorConfiguration()
        
        # Initialize analysis components
        self.uncertainty_scorer = uncertainty_scorer or UncertaintyScorer()
        self.bm25_gate = bm25_gate or BM25Gate()
        self.adaptive_k_selector = adaptive_k_selector or AdaptiveKSelector()
        
        # Initialize processing components
        self.rrf_merger = rrf_merger or RRFMerger(
            k=getattr(settings, "rrf_k", 60),
            normalize_scores=getattr(settings, "rrf_normalize_scores", True),
            score_threshold=self.config.rrf_score_threshold
        )
        self.deduplicator = deduplicator or DocumentDeduplicator(
            jaccard_threshold=0.82,  # Conservative thresholds for accuracy
            hamming_threshold=4
        )
        self.l2_cache = l2_cache
        
        # Store retriever functions
        self.retrievers = {
            "dense": dense_retriever,
            "sparse": sparse_retriever,
            "bm25": bm25_retriever,
            "hybrid": hybrid_retriever
        }
        
        # Thread pool for parallel execution
        self.executor = ThreadPoolExecutor(
            max_workers=self.config.max_parallel_workers,
            thread_name_prefix="retrieval"
        ) if self.config.enable_parallel_execution else None
        
        # Execution statistics
        self._total_queries = 0
        self._cache_hits = 0
        self._average_execution_time_ms = 0.0
        
        logger.info(f"GatedRetrievalCoordinator initialized with {len([r for r in self.retrievers.values() if r])} retrievers")
    
    async def retrieve(
        self, 
        query: str,
        max_final_docs: Optional[int] = None,
        force_cache_refresh: bool = False
    ) -> Tuple[List[Document], RetrievalCoordinationMetrics]:
        """
        Execute coordinated retrieval for the given query.
        
        Args:
            query: Search query
            max_final_docs: Maximum documents to return (None = use K-selector decision)
            force_cache_refresh: Skip cache lookup and force fresh retrieval
            
        Returns:
            Tuple of (final_documents, coordination_metrics)
        """
        overall_start = time.time()
        
        try:
            # Update statistics
            self._total_queries += 1
            
            # Phase 1: Query Analysis
            logger.debug(f"Starting coordinated retrieval for: '{query[:50]}...'")
            
            analysis_start = time.time()
            
            # Uncertainty analysis
            uncertainty_start = time.time()
            uncertainty_result = self.uncertainty_scorer.score_query(query)
            uncertainty_time = (time.time() - uncertainty_start) * 1000
            
            # BM25 gating decision
            bm25_start = time.time()
            bm25_result = self.bm25_gate.should_activate_bm25(query)
            bm25_time = (time.time() - bm25_start) * 1000
            
            # Adaptive K selection
            k_start = time.time()
            k_result = self.adaptive_k_selector.select_k(query)
            k_time = (time.time() - k_start) * 1000
            
            analysis_time = (time.time() - analysis_start) * 1000
            
            logger.debug(f"Analysis phase: uncertainty={uncertainty_result.confidence_level}, "
                        f"bm25_activate={bm25_result.should_activate}, "
                        f"complexity={k_result.query_complexity.value}, "
                        f"total_k={k_result.k_profile.total_k}")
            
            # Phase 2: Cache Lookup
            cache_start = time.time()
            cached_result = None
            
            if (self.config.enable_l2_cache and self.l2_cache and 
                not force_cache_refresh):
                
                try:
                    cached_result = await self.l2_cache.get(
                        query=query,
                        retriever_names=k_result.k_profile.active_retrievers,
                        rrf_k=self.rrf_merger.k,
                        dedup_params={
                            "jaccard_threshold": self.deduplicator.jaccard_threshold,
                            "hamming_threshold": self.deduplicator.hamming_threshold
                        },
                        max_docs=k_result.k_profile.total_k
                    )
                except Exception as e:
                    logger.warning(f"Cache lookup error: {e}")
                    cached_result = None
            
            cache_time = (time.time() - cache_start) * 1000
            cache_hit = cached_result is not None
            
            if cache_hit:
                cached_rrf_docs = cached_result[0]
                
                # Apply score threshold to cached documents if needed
                if self.rrf_merger.score_threshold > 0.0 and cached_rrf_docs:
                    filtered_cached_docs = [
                        doc for doc in cached_rrf_docs
                        if doc.rrf_score >= self.rrf_merger.score_threshold
                    ]
                    if not filtered_cached_docs:
                        filtered_cached_docs = [
                            max(cached_rrf_docs, key=lambda doc: doc.rrf_score)
                        ]
                    cached_rrf_docs = filtered_cached_docs
                
                # Ensure RRF metadata is present on cached documents
                for rank, rrf_doc in enumerate(cached_rrf_docs):
                    metadata = dict(rrf_doc.document.metadata or {})
                    metadata["rrf_score"] = rrf_doc.rrf_score
                    metadata["rrf_rank"] = rank
                    metadata["rrf_retriever_ranks"] = rrf_doc.retriever_ranks
                    metadata["rrf_retriever_scores"] = rrf_doc.retriever_scores
                    metadata["rrf_retrievers"] = list(rrf_doc.retriever_ranks.keys())
                    rrf_doc.document.metadata = metadata
                
                logger.debug(f"Cache hit! Returning {len(cached_rrf_docs)} cached documents")
                self._cache_hits += 1
                
                # Apply final document limit if specified
                final_docs = cached_rrf_docs
                if max_final_docs and len(final_docs) > max_final_docs:
                    final_docs = final_docs[:max_final_docs]
                
                # Convert RRFDocuments back to Documents
                final_documents = [rrf_doc.document for rrf_doc in final_docs]
                
                # Create metrics for cache hit
                metrics = self._create_cache_hit_metrics(
                    query, uncertainty_result, k_result, cached_result[1],
                    uncertainty_time, bm25_time, k_time, cache_time,
                    len(final_documents), overall_start
                )
                
                return final_documents, metrics
            
            # Phase 3: Parallel Retrieval Execution  
            retrieval_start = time.time()
            
            retrieval_results = await self._execute_parallel_retrieval(
                query, k_result.k_profile, k_result.retriever_ks
            )
            
            retrieval_time = (time.time() - retrieval_start) * 1000
            
            # Check if we got any results
            total_retrieved = sum(len(result.documents) for result in retrieval_results.values())
            if total_retrieved == 0:
                logger.warning(f"No documents retrieved for query: '{query}'")
                return [], self._create_empty_metrics(query, uncertainty_result, overall_start)
            
            # Phase 4: RRF Merging
            merge_start = time.time()
            
            retriever_docs = {
                name: result.documents 
                for name, result in retrieval_results.items()
                if result.success and result.documents
            }
            
            merged_docs, rrf_stats = self.rrf_merger.merge(retriever_docs)
            merge_time = (time.time() - merge_start) * 1000
            
            logger.debug(f"RRF merged {len(merged_docs)} documents from {len(retriever_docs)} retrievers")
            
            # Phase 5: Deduplication
            dedup_start = time.time()
            
            # Convert RRFDocuments to Documents for deduplication
            docs_for_dedup = [rrf_doc.document for rrf_doc in merged_docs]
            unique_documents, dedup_stats = self.deduplicator.deduplicate(docs_for_dedup)
            
            dedup_time = (time.time() - dedup_start) * 1000
            
            logger.debug(f"Deduplication: {len(docs_for_dedup)} → {len(unique_documents)} "
                        f"(ratio: {dedup_stats.get_deduplication_ratio():.2f})")
            
            # Phase 6: Final Document Selection
            final_documents = unique_documents
            if max_final_docs and len(final_documents) > max_final_docs:
                final_documents = final_documents[:max_final_docs]
            
            # Phase 7: Cache Storage
            cache_store_start = time.time()
            cache_stored = False
            
            if (self.config.enable_l2_cache and self.l2_cache and 
                len(merged_docs) >= self.config.cache_threshold_docs):
                
                try:
                    cache_stored = await self.l2_cache.set(
                        query=query,
                        retriever_names=k_result.k_profile.active_retrievers,
                        rrf_k=self.rrf_merger.k,
                        dedup_params={
                            "jaccard_threshold": self.deduplicator.jaccard_threshold,
                            "hamming_threshold": self.deduplicator.hamming_threshold
                        },
                        rrf_documents=merged_docs,
                        retriever_stats={
                            "total_retrieved": total_retrieved,
                            "execution_time_ms": retrieval_time,
                            "retrievers": list(retrieval_results.keys())
                        },
                        max_docs=k_result.k_profile.total_k
                    )
                except Exception as e:
                    logger.warning(f"Cache storage error: {e}")
            
            cache_store_time = (time.time() - cache_store_start) * 1000
            
            # Create comprehensive metrics
            total_time = (time.time() - overall_start) * 1000
            
            metrics = RetrievalCoordinationMetrics(
                query=query,
                total_execution_time_ms=total_time,
                
                # Analysis phase
                uncertainty_analysis_time_ms=uncertainty_time,
                bm25_gating_time_ms=bm25_time,
                k_selection_time_ms=k_time,
                
                # Execution phase
                cache_lookup_time_ms=cache_time,
                retrieval_execution_time_ms=retrieval_time,
                rrf_merge_time_ms=merge_time,
                deduplication_time_ms=dedup_time,
                cache_store_time_ms=cache_store_time,
                
                # Results
                retrievers_executed=list(retrieval_results.keys()),
                total_documents_retrieved=total_retrieved,
                unique_documents_after_dedup=len(unique_documents),
                final_documents_returned=len(final_documents),
                
                # Performance
                cache_hit=False,
                deduplication_ratio=dedup_stats.get_deduplication_ratio() / 100.0,
                average_retrieval_latency_ms=sum(
                    r.execution_time_ms for r in retrieval_results.values()
                ) / max(1, len(retrieval_results)),
                
                # Quality
                uncertainty_score=uncertainty_result.overall_uncertainty,
                estimated_recall_coverage=k_result.estimated_recall_coverage,
                rrf_merger_stats=rrf_stats,
                deduplication_stats=dedup_stats.__dict__ if dedup_stats else None
            )
            
            # Update running statistics
            self._update_running_stats(total_time)
            
            logger.info(f"Coordinated retrieval completed: {len(final_documents)} docs in {total_time:.1f}ms "
                       f"(cache: {'hit' if cache_hit else 'miss'}, "
                       f"dedup: {dedup_stats.get_deduplication_ratio():.1f}%)")
            
            return final_documents, metrics
            
        except Exception as e:
            logger.error(f"Error in coordinated retrieval: {e}", exc_info=True)
            
            # Return empty result with error metrics
            total_time = (time.time() - overall_start) * 1000
            error_metrics = self._create_error_metrics(query, str(e), total_time)
            return [], error_metrics
    
    async def _execute_parallel_retrieval(
        self,
        query: str,
        k_profile,
        retriever_ks: Dict[str, int]
    ) -> Dict[str, RetrieverExecutionResult]:
        """Execute retrieval across all active retrievers in parallel."""
        
        # Determine which retrievers to execute
        active_retrievers = [
            (name, k_val) for name, k_val in retriever_ks.items()
            if k_val > 0 and self.retrievers.get(name) is not None
        ]
        
        if not active_retrievers:
            logger.warning("No active retrievers available")
            return {}
        
        logger.debug(f"Executing {len(active_retrievers)} retrievers: {[name for name, _ in active_retrievers]}")
        
        # Execute retrievers in parallel
        retrieval_tasks = []
        
        for retriever_name, k_value in active_retrievers:
            task = self._execute_single_retriever(
                retriever_name, self.retrievers[retriever_name], query, k_value
            )
            retrieval_tasks.append(task)
        
        # Wait for all retrievers to complete
        results = {}
        
        if self.config.enable_parallel_execution and self.executor:
            # Use asyncio gather for parallel execution
            try:
                retrieval_results = await asyncio.wait_for(
                    asyncio.gather(*retrieval_tasks, return_exceptions=True),
                    timeout=self.config.retrieval_timeout_ms / 1000.0
                )
                
                for i, result in enumerate(retrieval_results):
                    retriever_name = active_retrievers[i][0]
                    if isinstance(result, Exception):
                        logger.error(f"Retriever {retriever_name} failed: {result}")
                        results[retriever_name] = RetrieverExecutionResult(
                            retriever_name=retriever_name,
                            documents=[],
                            execution_time_ms=0.0,
                            documents_requested=active_retrievers[i][1],
                            documents_returned=0,
                            success=False,
                            error_message=str(result)
                        )
                    else:
                        results[retriever_name] = result
                        
            except asyncio.TimeoutError:
                logger.error(f"Retrieval timeout after {self.config.retrieval_timeout_ms}ms")
                # Create timeout results for all retrievers
                for retriever_name, k_value in active_retrievers:
                    results[retriever_name] = RetrieverExecutionResult(
                        retriever_name=retriever_name,
                        documents=[],
                        execution_time_ms=self.config.retrieval_timeout_ms,
                        documents_requested=k_value,
                        documents_returned=0,
                        success=False,
                        error_message="Timeout"
                    )
                    
        else:
            # Sequential execution fallback
            for task in retrieval_tasks:
                try:
                    result = await task
                    results[result.retriever_name] = result
                except Exception as e:
                    retriever_name = active_retrievers[len(results)][0]
                    logger.error(f"Retriever {retriever_name} failed: {e}")
                    results[retriever_name] = RetrieverExecutionResult(
                        retriever_name=retriever_name,
                        documents=[],
                        execution_time_ms=0.0,
                        documents_requested=active_retrievers[len(results)][1],
                        documents_returned=0,
                        success=False,
                        error_message=str(e)
                    )
        
        return results
    
    async def _execute_single_retriever(
        self,
        retriever_name: str,
        retriever_func: callable,
        query: str,
        k: int
    ) -> RetrieverExecutionResult:
        """Execute a single retriever and return formatted result."""
        
        start_time = time.time()
        
        try:
            logger.debug(f"Executing {retriever_name} retriever with k={k}")
            
            # Execute retriever (assume it returns List[Document])
            if asyncio.iscoroutinefunction(retriever_func):
                documents = await retriever_func(query, k)
            else:
                # Run synchronous retriever in thread pool
                loop = asyncio.get_event_loop()
                documents = await loop.run_in_executor(
                    self.executor, retriever_func, query, k
                )
            
            execution_time = (time.time() - start_time) * 1000
            
            # Validate results
            if not isinstance(documents, list):
                raise ValueError(f"Retriever {retriever_name} returned non-list: {type(documents)}")
            
            # Ensure all items are Documents
            valid_docs = []
            for doc in documents:
                if isinstance(doc, Document):
                    valid_docs.append(doc)
                else:
                    logger.warning(f"Retriever {retriever_name} returned non-Document: {type(doc)}")
            
            logger.debug(f"Retriever {retriever_name} returned {len(valid_docs)} documents in {execution_time:.1f}ms")
            
            return RetrieverExecutionResult(
                retriever_name=retriever_name,
                documents=valid_docs,
                execution_time_ms=execution_time,
                documents_requested=k,
                documents_returned=len(valid_docs),
                success=True
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"Retriever {retriever_name} failed: {e}")
            
            return RetrieverExecutionResult(
                retriever_name=retriever_name,
                documents=[],
                execution_time_ms=execution_time,
                documents_requested=k,
                documents_returned=0,
                success=False,
                error_message=str(e)
            )
    
    def _create_cache_hit_metrics(
        self, query: str, uncertainty_result: UncertaintyResult, 
        k_result: AdaptiveKResult, retriever_stats: Dict[str, Any],
        uncertainty_time: float, bm25_time: float, k_time: float, 
        cache_time: float, final_doc_count: int, overall_start: float
    ) -> RetrievalCoordinationMetrics:
        """Create metrics for cache hit scenario."""
        
        total_time = (time.time() - overall_start) * 1000
        
        return RetrievalCoordinationMetrics(
            query=query,
            total_execution_time_ms=total_time,
            uncertainty_analysis_time_ms=uncertainty_time,
            bm25_gating_time_ms=bm25_time,
            k_selection_time_ms=k_time,
            cache_lookup_time_ms=cache_time,
            retrieval_execution_time_ms=0.0,  # Cache hit
            rrf_merge_time_ms=0.0,  # Cache hit
            deduplication_time_ms=0.0,  # Cache hit
            cache_store_time_ms=0.0,  # Cache hit
            retrievers_executed=[],  # Cache hit
            total_documents_retrieved=0,  # Cache hit
            unique_documents_after_dedup=final_doc_count,
            final_documents_returned=final_doc_count,
            cache_hit=True,
            deduplication_ratio=0.0,  # Unknown for cache hit
            average_retrieval_latency_ms=0.0,
            uncertainty_score=uncertainty_result.overall_uncertainty,
            estimated_recall_coverage=k_result.estimated_recall_coverage
        )
    
    def _create_empty_metrics(
        self, query: str, uncertainty_result: UncertaintyResult, overall_start: float
    ) -> RetrievalCoordinationMetrics:
        """Create metrics for empty result scenario."""
        
        total_time = (time.time() - overall_start) * 1000
        
        return RetrievalCoordinationMetrics(
            query=query,
            total_execution_time_ms=total_time,
            uncertainty_analysis_time_ms=0.0,
            bm25_gating_time_ms=0.0,
            k_selection_time_ms=0.0,
            cache_lookup_time_ms=0.0,
            retrieval_execution_time_ms=0.0,
            rrf_merge_time_ms=0.0,
            deduplication_time_ms=0.0,
            cache_store_time_ms=0.0,
            retrievers_executed=[],
            total_documents_retrieved=0,
            unique_documents_after_dedup=0,
            final_documents_returned=0,
            cache_hit=False,
            deduplication_ratio=0.0,
            average_retrieval_latency_ms=0.0,
            uncertainty_score=uncertainty_result.overall_uncertainty,
            estimated_recall_coverage=0.0
        )
    
    def _create_error_metrics(
        self, query: str, error_message: str, total_time: float
    ) -> RetrievalCoordinationMetrics:
        """Create metrics for error scenario."""
        
        return RetrievalCoordinationMetrics(
            query=query,
            total_execution_time_ms=total_time,
            uncertainty_analysis_time_ms=0.0,
            bm25_gating_time_ms=0.0,
            k_selection_time_ms=0.0,
            cache_lookup_time_ms=0.0,
            retrieval_execution_time_ms=0.0,
            rrf_merge_time_ms=0.0,
            deduplication_time_ms=0.0,
            cache_store_time_ms=0.0,
            retrievers_executed=[],
            total_documents_retrieved=0,
            unique_documents_after_dedup=0,
            final_documents_returned=0,
            cache_hit=False,
            deduplication_ratio=0.0,
            average_retrieval_latency_ms=0.0,
            uncertainty_score=1.0,  # Maximum uncertainty for errors
            estimated_recall_coverage=0.0
        )
    
    def _update_running_stats(self, execution_time_ms: float) -> None:
        """Update running execution statistics."""
        if self._total_queries == 1:
            self._average_execution_time_ms = execution_time_ms
        else:
            # Running average
            self._average_execution_time_ms = (
                (self._average_execution_time_ms * (self._total_queries - 1) + execution_time_ms) 
                / self._total_queries
            )
    
    def get_coordinator_stats(self) -> Dict[str, Any]:
        """Get coordinator performance statistics."""
        return {
            "total_queries": self._total_queries,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": self._cache_hits / max(1, self._total_queries),
            "average_execution_time_ms": self._average_execution_time_ms,
            "configuration": {
                "enable_l2_cache": self.config.enable_l2_cache,
                "enable_parallel_execution": self.config.enable_parallel_execution,
                "max_parallel_workers": self.config.max_parallel_workers,
                "retrieval_timeout_ms": self.config.retrieval_timeout_ms
            },
            "active_retrievers": list(name for name, func in self.retrievers.items() if func),
            "component_status": {
                "uncertainty_scorer": self.uncertainty_scorer is not None,
                "bm25_gate": self.bm25_gate is not None,
                "adaptive_k_selector": self.adaptive_k_selector is not None,
                "rrf_merger": self.rrf_merger is not None,
                "deduplicator": self.deduplicator is not None,
                "l2_cache": self.l2_cache is not None
            }
        }
    
    def close(self) -> None:
        """Clean up resources."""
        if self.executor:
            self.executor.shutdown(wait=True)
        logger.info("GatedRetrievalCoordinator closed")


def create_gated_retrieval_coordinator(
    # Retriever functions
    dense_retriever: Optional[callable] = None,
    sparse_retriever: Optional[callable] = None,
    bm25_retriever: Optional[callable] = None,
    hybrid_retriever: Optional[callable] = None,
    
    # Components (will use defaults if not provided)
    uncertainty_scorer: Optional[UncertaintyScorer] = None,
    bm25_gate: Optional[BM25Gate] = None,
    adaptive_k_selector: Optional[AdaptiveKSelector] = None,
    rrf_merger: Optional[RRFMerger] = None,
    deduplicator: Optional[DocumentDeduplicator] = None,
    l2_cache: Optional[RetrievalL2Cache] = None,
    
    # Configuration
    config: Optional[CoordinatorConfiguration] = None
) -> GatedRetrievalCoordinator:
    """
    Factory function to create a gated retrieval coordinator.
    
    Args:
        dense_retriever: Semantic/dense retrieval function
        sparse_retriever: Sparse/lexical retrieval function  
        bm25_retriever: BM25 keyword retrieval function
        hybrid_retriever: Hybrid retrieval function
        uncertainty_scorer: Custom uncertainty scorer
        bm25_gate: Custom BM25 gate
        adaptive_k_selector: Custom K selector
        rrf_merger: Custom RRF merger
        deduplicator: Custom deduplicator
        l2_cache: Custom L2 cache
        config: Coordinator configuration
        
    Returns:
        Configured GatedRetrievalCoordinator instance
    """
    return GatedRetrievalCoordinator(
        uncertainty_scorer=uncertainty_scorer,
        bm25_gate=bm25_gate,
        adaptive_k_selector=adaptive_k_selector,
        rrf_merger=rrf_merger,
        deduplicator=deduplicator,
        l2_cache=l2_cache,
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        bm25_retriever=bm25_retriever,
        hybrid_retriever=hybrid_retriever,
        config=config
    )
