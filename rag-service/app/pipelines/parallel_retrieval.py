"""Parallel retrieval pipeline for concurrent retriever execution."""

import asyncio
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime
import time

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models import BaseLLM

from app.core.logging import get_logger
from app.core.config import settings
from app.components.reranker import CrossEncoderReranker
from app.components.rrf_merger import RRFMerger
from app.components.table_ranker import TableRanker
from app.unified_retrieval.unified_retriever import UnifiedRetriever
from app.services.performance_monitor import get_performance_monitor

logger = get_logger(__name__)

_FLOAT_EPSILON = 1e-9


class CircuitBreaker:
    """Circuit breaker for failing retrievers."""
    
    def __init__(self, failure_threshold: int = 3, timeout: float = 60.0):
        """Initialize circuit breaker."""
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failures: Dict[str, int] = {}
        self.last_failure_time: Dict[str, float] = {}
        
    def is_open(self, retriever_name: str) -> bool:
        """Check if circuit is open (retriever should be skipped)."""
        if retriever_name not in self.failures:
            return False
            
        if self.failures[retriever_name] >= self.failure_threshold:
            # Check if timeout has passed
            if time.time() - self.last_failure_time[retriever_name] > self.timeout:
                # Reset circuit
                self.failures[retriever_name] = 0
                return False
            return True
        return False
        
    def record_failure(self, retriever_name: str):
        """Record a failure for a retriever."""
        self.failures[retriever_name] = self.failures.get(retriever_name, 0) + 1
        self.last_failure_time[retriever_name] = time.time()
        
    def record_success(self, retriever_name: str):
        """Record a success for a retriever."""
        if retriever_name in self.failures:
            self.failures[retriever_name] = 0


class ParallelRetrievalPipeline:
    """Pipeline for executing multiple retrievers in parallel."""
    
    def __init__(
        self,
        retrievers: Dict[str, BaseRetriever],
        weights: Optional[Dict[str, float]] = None,
        concurrency_limit: int = 5,
        timeout_per_retriever: float = 10.0,
        circuit_breaker: Optional[CircuitBreaker] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        table_ranker: Optional[TableRanker] = None,
        rrf_merger: Optional[RRFMerger] = None
    ):
        """
        Initialize parallel retrieval pipeline.
        
        Args:
            retrievers: Dictionary of retriever name to retriever instance
            weights: Optional weights for each retriever (for result merging)
            concurrency_limit: Maximum number of concurrent retriever calls
            timeout_per_retriever: Timeout for each retriever in seconds
            circuit_breaker: Optional circuit breaker for handling failures
            reranker: Optional CrossEncoderReranker for result reranking
            table_ranker: Optional TableRanker for table-specific queries
            rrf_merger: Optional RRFMerger for result fusion and thresholding
        """
        self.retrievers = retrievers
        self.weights = weights or {name: 1.0 for name in retrievers}
        self.concurrency_limit = concurrency_limit
        self.timeout_per_retriever = timeout_per_retriever
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self.reranker = reranker
        self.table_ranker = table_ranker
        self.rrf_merger = rrf_merger or RRFMerger(
            k=getattr(settings, "rrf_k", 60),
            normalize_scores=getattr(settings, "rrf_normalize_scores", True),
            score_threshold=getattr(settings, "rrf_score_threshold", 0.0)
        )
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        if total_weight == 0:
            total_weight = 1.0
        self.weights = {k: v / total_weight for k, v in self.weights.items()}

    def _get_document_key(self, doc: Document) -> str:
        """Return a deterministic identifier for a document."""
        metadata = getattr(doc, "metadata", {}) or {}
        if hasattr(metadata, "model_dump"):
            metadata = metadata.model_dump()
        if not isinstance(metadata, dict):
            metadata = {}

        for field in (
            "id",
            "chunk_id",
            "document_id",
            "parent_id",
            "source_id",
            "page_id",
            "uid",
        ):
            value = metadata.get(field)
            if value:
                return f"{field}:{value}"

        content = doc.page_content or ""
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def _should_replace_candidate(
        self,
        existing: Dict[str, Any],
        candidate: Dict[str, Any]
    ) -> bool:
        """Determine if candidate should replace existing document entry."""
        if candidate["score"] > existing["score"] + _FLOAT_EPSILON:
            return True
        if existing["score"] > candidate["score"] + _FLOAT_EPSILON:
            return False

        if candidate["weight"] > existing["weight"] + _FLOAT_EPSILON:
            return True
        if existing["weight"] > candidate["weight"] + _FLOAT_EPSILON:
            return False

        if candidate["position"] < existing["position"]:
            return True
        if candidate["position"] > existing["position"]:
            return False

        if candidate["retriever"] < existing["retriever"]:
            return True
        if candidate["retriever"] > existing["retriever"]:
            return False

        return candidate["key"] < existing["key"]
        
    async def _retrieve_with_timeout(
        self,
        retriever_name: str,
        retriever: BaseRetriever,
        query: str,
        k: int
    ) -> Tuple[str, List[Document], float]:
        """Retrieve documents with timeout."""
        start_time = time.time()
        
        try:
            # Check if it's a UnifiedRetriever and pass k parameter
            if isinstance(retriever, UnifiedRetriever):
                # UnifiedRetriever supports passing k directly
                docs = await asyncio.wait_for(
                    retriever.aget_relevant_documents(query, top_k=k),
                    timeout=self.timeout_per_retriever
                )
            else:
                # Standard retriever - use ainvoke (LangChain's new API)
                docs = await asyncio.wait_for(
                    retriever.ainvoke(query),
                    timeout=self.timeout_per_retriever
                )
            
            elapsed = time.time() - start_time
            
            # Record success
            self.circuit_breaker.record_success(retriever_name)
            
            # Log metrics for UnifiedRetriever
            if isinstance(retriever, UnifiedRetriever):
                metrics = retriever.get_pipeline_metrics()
                logger.info(f"UnifiedRetriever {retriever_name} metrics: {metrics}")
            
            # Limit results to k
            return retriever_name, docs[:k], elapsed
            
        except asyncio.TimeoutError:
            logger.warning(f"Retriever {retriever_name} timed out after {self.timeout_per_retriever}s")
            self.circuit_breaker.record_failure(retriever_name)
            return retriever_name, [], self.timeout_per_retriever
            
        except Exception as e:
            logger.error(f"Retriever {retriever_name} failed: {e}")
            self.circuit_breaker.record_failure(retriever_name)
            elapsed = time.time() - start_time
            return retriever_name, [], elapsed
    
    async def retrieve(
        self,
        query: str,
        k: int = 5,
        merge_strategy: str = "weighted",
        hyde_hypothesis: Optional[str] = None,
        hyde_generator: Optional[Any] = None,
        auxiliary_model: Optional[str] = None,
        auxiliary_provider: Optional[Any] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve documents from all retrievers in parallel.

        Args:
            query: Search query
            k: Number of documents to return
            merge_strategy: How to merge results ("weighted", "round_robin", "score_based")
            hyde_hypothesis: Optional HyDE hypothetical document string
            hyde_generator: Optional HyDE generator instance for concurrent generation
            auxiliary_model: Optional model to use for HyDE generation
            auxiliary_provider: Optional provider to use for HyDE generation

        Returns:
            List of (document, score) tuples
        """
        # Filter out retrievers with open circuits
        active_retrievers = {
            name: retriever
            for name, retriever in self.retrievers.items()
            if not self.circuit_breaker.is_open(name)
        }

        if not active_retrievers:
            logger.warning("All retrievers have open circuits!")
            return []
            
        monitor = get_performance_monitor()
        start_time = time.perf_counter()

        # 1. Start standard retrieval tasks immediately
        standard_tasks = []
        for name, retriever in active_retrievers.items():
            task = self._retrieve_with_timeout(name, retriever, query, k * 2)
            standard_tasks.append(task)
            
        # 2. Start HyDE generation in background if generator provided
        hyde_gen_task = None
        if hyde_generator and not hyde_hypothesis and settings.enable_hyde:
            hyde_gen_task = asyncio.create_task(hyde_generator.generate_hypothesis(query, model=auxiliary_model, provider=auxiliary_provider))

        # 3. Execute Standard Tasks
        results_by_retriever = {}
        latencies = {}
        
        # Process standard tasks
        if standard_tasks:
            # Gather standard results
            batch_results = await asyncio.gather(*standard_tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Retrieval task failed: {result}")
                else:
                    name, docs, latency = result
                    if docs:
                        results_by_retriever[name] = docs
                    latencies[name] = latency
                    if monitor:
                        try:
                            monitor.record_retriever_performance(name, float(latency) * 1000, len(docs))
                        except Exception:
                            pass

        # 4. Check Early Exit / Execute HyDE
        # Determine if we should proceed with HyDE based on standard results
        should_run_hyde = False
        final_hypothesis = hyde_hypothesis

        # If we explicitly have a hypothesis string, we always run HyDE
        if hyde_hypothesis:
            should_run_hyde = True
        # If we have a generator task, check if we need it
        elif hyde_gen_task:
            # Check standard results confidence
            max_score = 0.0
            total_docs = 0
            
            # Extract scores if available (depends on retriever type)
            for docs in results_by_retriever.values():
                total_docs += len(docs)
                for doc in docs:
                    # Try to find a relevance score
                    if hasattr(doc, "metadata") and "score" in doc.metadata:
                        try:
                            max_score = max(max_score, float(doc.metadata["score"]))
                        except (ValueError, TypeError):
                            pass
            
            # Early Exit Condition: High confidence results found
            # Note: Vector store scores vary. Assuming standard normalized scores or strong keyword matches.
            # If we have plenty of results, we might skip HyDE for speed
            if total_docs >= k * 2 and max_score > 0.88:
                logger.info(f"Early exit: High confidence standard results (score={max_score:.2f}, count={total_docs}). Skipping HyDE.")
                should_run_hyde = False
                # Cancel the generation task to save LLM tokens/slots
                hyde_gen_task.cancel()
            else:
                should_run_hyde = True
                try:
                    # Await the hypothesis generation
                    gen_start = time.perf_counter()
                    final_hypothesis = await hyde_gen_task
                    gen_time = (time.perf_counter() - gen_start) * 1000
                    if monitor:
                        monitor.record_latency("hyde_latency_ms", gen_time)
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.warning(f"HyDE generation failed in background: {e}")

        # 5. Execute HyDE Retrieval if needed
        if should_run_hyde and final_hypothesis and final_hypothesis != query:
            logger.info("Executing concurrent HyDE retrieval")
            hyde_tasks = []
            # Only use vector retrievers for HyDE
            for name, retriever in active_retrievers.items():
                if "bm25" not in name.lower():
                    hyde_task = self._retrieve_with_timeout(
                        f"{name}_hyde", retriever, final_hypothesis, k
                    )
                    hyde_tasks.append(hyde_task)
            
            if hyde_tasks:
                hyde_results = await asyncio.gather(*hyde_tasks, return_exceptions=True)
                for result in hyde_results:
                    if isinstance(result, Exception):
                        logger.error(f"HyDE retrieval task failed: {result}")
                    else:
                        name, docs, latency = result
                        if docs:
                            results_by_retriever[name] = docs
                        latencies[name] = latency

        
        # Log retrieval metrics
        logger.info(f"Parallel retrieval completed - Active: {len(active_retrievers)}, "
                   f"Successful: {len(results_by_retriever)}, "
                   f"Latencies: {latencies}")
        
        merged_results: List[Tuple[Document, float]] = []
        
        if self.rrf_merger and results_by_retriever:
            rrf_start = time.perf_counter()
            max_docs = k * 2 if self.reranker else k
            rrf_docs, rrf_stats = self.rrf_merger.merge(results_by_retriever, max_docs=max_docs)
            merged_results = [(rrf_doc.document, rrf_doc.rrf_score) for rrf_doc in rrf_docs]
            
            if monitor:
                monitor.record_latency("rrf_merge_latency_ms", (time.perf_counter() - rrf_start) * 1000)

            logger.debug(
                "RRF merge completed with %d docs (filtered=%d, threshold=%.2f)",
                len(merged_results),
                rrf_stats.filtered_below_threshold if rrf_stats else 0,
                self.rrf_merger.score_threshold
            )
        else:
            merged_results = self._merge_results(
                results_by_retriever,
                k * 2 if self.reranker else k,
                merge_strategy
            )
        
        query_lower = query.lower()
        is_table_query = any(
            term in query_lower
            for term in ["rate", "allowance", "table", "$", "meal", "incidental", "kilometric", "per km"]
        )
        is_trip_planning = any(term in query_lower for term in ["trip", "travel", "journey", "planning"])
        needs_cost_info = any(
            term in query_lower for term in ["cost", "expense", "estimate", "budget", "how much"]
        )

        if is_trip_planning and needs_cost_info:
            is_table_query = True
            logger.info("Detected trip planning query with cost estimation - applying table ranking")

        table_ranker_applied = False

        if is_table_query and self.table_ranker and merged_results:
            logger.info("Applying table-specific ranking (pre-reranker)")
            table_start = time.perf_counter()
            score_map = {self._get_document_key(doc): score for doc, score in merged_results}
            documents_for_ranking = [doc for doc, _ in merged_results]
            ranked_docs = self.table_ranker.filter_and_rerank(
                documents_for_ranking,
                query,
                top_k=len(documents_for_ranking),
                query_type="table",
            )
            merged_results = [
                (doc, score_map.get(self._get_document_key(doc), 1.0 - (idx * 0.01)))
                for idx, doc in enumerate(ranked_docs)
            ]
            if monitor:
                monitor.record_latency("table_ranker_latency_ms", (time.perf_counter() - table_start) * 1000)
            table_ranker_applied = True

        # Apply reranking if available
        if self.reranker and merged_results:
            # Extract documents from tuples
            documents = [doc for doc, _ in merged_results]

            if is_table_query and self.table_ranker and not table_ranker_applied:
                logger.info("Applying table-specific ranking")
                table_start = time.perf_counter()
                documents = self.table_ranker.filter_and_rerank(
                    documents,
                    query,
                    top_k=min(len(documents), k * 2),  # Keep more for final reranking
                    query_type="table",
                )
                if monitor:
                    monitor.record_latency("table_ranker_latency_ms", (time.perf_counter() - table_start) * 1000)
                table_ranker_applied = True

            # Apply general reranking
            logger.info(f"Applying reranking to {len(documents)} documents")
            rerank_start = time.perf_counter()
            reranked_docs = await self.reranker.arerank(query, documents, k)
            if monitor:
                monitor.record_latency("reranker_latency_ms", (time.perf_counter() - rerank_start) * 1000)
            
            # Convert back to tuples with scores
            merged_results = [(doc, 1.0 - (i * 0.1)) for i, doc in enumerate(reranked_docs)]
            logger.info(f"Reranking complete, returning {len(merged_results)} documents")
        
        return merged_results
    
    def _merge_results(
        self,
        results_by_retriever: Dict[str, List[Document]],
        k: int,
        strategy: str
    ) -> List[Tuple[Document, float]]:
        """Merge results from multiple retrievers."""
        
        if strategy == "weighted":
            return self._weighted_merge(results_by_retriever, k)
        elif strategy == "round_robin":
            return self._round_robin_merge(results_by_retriever, k)
        elif strategy == "score_based":
            return self._score_based_merge(results_by_retriever, k)
        else:
            logger.warning(f"Unknown merge strategy: {strategy}, using weighted")
            return self._weighted_merge(results_by_retriever, k)
    
    def _ordered_retriever_items(
        self,
        results_by_retriever: Dict[str, List[Document]]
    ) -> List[Tuple[str, List[Document]]]:
        """Return retriever results in deterministic priority order."""
        return sorted(
            results_by_retriever.items(),
            key=lambda item: (-self.weights.get(item[0], 0.0), item[0])
        )
    
    def _weighted_merge(
        self,
        results_by_retriever: Dict[str, List[Document]],
        k: int
    ) -> List[Tuple[Document, float]]:
        """Merge results using weighted scores."""
        merged: Dict[str, Dict[str, Any]] = {}
        
        for retriever_name, docs in self._ordered_retriever_items(results_by_retriever):
            weight = self.weights.get(retriever_name, 1.0)
            
            for position, doc in enumerate(docs):
                key = self._get_document_key(doc)
                position_score = 1.0 / (position + 1)
                final_score = weight * position_score
                
                candidate = {
                    "doc": doc,
                    "score": final_score,
                    "weight": weight,
                    "position": position,
                    "retriever": retriever_name,
                    "key": key,
                }
                
                existing = merged.get(key)
                if existing is None or self._should_replace_candidate(existing, candidate):
                    merged[key] = candidate
        
        ordered = sorted(
            merged.values(),
            key=lambda item: (
                -item["score"],
                -item["weight"],
                item["position"],
                item["retriever"],
                item["key"],
            )
        )
        return [(item["doc"], item["score"]) for item in ordered[:k]]
    
    def _round_robin_merge(
        self,
        results_by_retriever: Dict[str, List[Document]],
        k: int
    ) -> List[Tuple[Document, float]]:
        """Merge results using round-robin selection."""
        seen_keys: Set[str] = set()
        merged_results: List[Tuple[Document, float]] = []
        
        ordered_items = self._ordered_retriever_items(results_by_retriever)
        iterators = {name: iter(docs) for name, docs in ordered_items}
        ordered_names = [name for name, _ in ordered_items]
        
        while len(merged_results) < k and ordered_names:
            exhausted: List[str] = []
            
            for name in ordered_names:
                iterator = iterators[name]
                while True:
                    try:
                        doc = next(iterator)
                    except StopIteration:
                        exhausted.append(name)
                        break
                    
                    key = self._get_document_key(doc)
                    if key in seen_keys:
                        # Try next document from the same retriever
                        continue
                    
                    seen_keys.add(key)
                    score = self.weights.get(name, 1.0)
                    merged_results.append((doc, score))
                    
                    if len(merged_results) >= k:
                        break
                    
                    # move to next retriever after recording a result
                    break
                
                if len(merged_results) >= k:
                    break
            
            # Remove exhausted retrievers from the rotation
            if exhausted:
                ordered_names = [name for name in ordered_names if name not in exhausted]
        
        # Deterministic ordering for returned results
        merged_results.sort(
            key=lambda item: (
                -item[1],
                self._get_document_key(item[0]),
            )
        )
        return merged_results[:k]
    
    def _score_based_merge(
        self,
        results_by_retriever: Dict[str, List[Document]],
        k: int
    ) -> List[Tuple[Document, float]]:
        """Merge results based on metadata scores if available."""
        merged: Dict[str, Dict[str, Any]] = {}
        
        for retriever_name, docs in self._ordered_retriever_items(results_by_retriever):
            weight = self.weights.get(retriever_name, 1.0)
            
            for position, doc in enumerate(docs):
                key = self._get_document_key(doc)
                
                metadata = getattr(doc, "metadata", {}) or {}
                if hasattr(metadata, "model_dump"):
                    metadata = metadata.model_dump()
                if not isinstance(metadata, dict):
                    metadata = {}
                
                metadata_score = metadata.get("score", 0.5)
                try:
                    metadata_score = float(metadata_score)
                except (TypeError, ValueError):
                    metadata_score = 0.5
                
                final_score = weight * metadata_score
                
                candidate = {
                    "doc": doc,
                    "score": final_score,
                    "weight": weight,
                    "position": position,
                    "retriever": retriever_name,
                    "key": key,
                }
                
                existing = merged.get(key)
                if existing is None or self._should_replace_candidate(existing, candidate):
                    merged[key] = candidate
        
        ordered = sorted(
            merged.values(),
            key=lambda item: (
                -item["score"],
                -item["weight"],
                item["position"],
                item["retriever"],
                item["key"],
            )
        )
        return [(item["doc"], item["score"]) for item in ordered[:k]]
    
    def get_retriever_stats(self) -> Dict[str, Any]:
        """Get statistics about retriever performance."""
        stats = {
            "retrievers": list(self.retrievers.keys()),
            "weights": self.weights,
            "circuit_breaker": {
                name: {
                    "failures": self.circuit_breaker.failures.get(name, 0),
                    "is_open": self.circuit_breaker.is_open(name)
                }
                for name in self.retrievers
            }
        }
        return stats


def create_parallel_pipeline(
    vector_store_manager,
    llm: Optional[BaseLLM] = None,
    retriever_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    enable_unified: bool = None,
    enable_reranker: bool = True,
    enable_stateful: bool = None,
    redis_client = None,
    rrf_k: Optional[int] = None,
) -> ParallelRetrievalPipeline:
    """Create a parallel retrieval pipeline with default retrievers."""
    from app.pipelines.retriever_factory import HybridRetrieverFactory, RetrieverConfig, RetrieverMode
    from app.unified_retrieval.migration import create_example_unified_config
    
    # Check if unified retrieval should be enabled from environment
    if enable_unified is None:
        enable_unified = settings.enable_unified_retrieval
    
    logger.info(f"Creating parallel pipeline - enable_unified: {enable_unified}")
    
    # Default retriever configurations
    if retriever_configs is None:
        retriever_configs = {
            "vector_similarity": {
                "type": "vector",
                "search_type": "similarity",
                "k": 10
            },
            "vector_mmr": {
                "type": "vector", 
                "search_type": "mmr",
                "k": 10,
                "lambda_mult": 0.5
            },
            "bm25": {
                "type": "bm25",
                "k": 10
            }
        }
        
        # Add multi-query if LLM is available
        if llm:
            retriever_configs["multi_query"] = {
                "type": "multi_query",
                "base_retriever": "vector_similarity",
                "llm": llm
            }
            
        # Add unified retriever if enabled
        if enable_unified:
            logger.info("Adding unified retriever to configuration")
            # Create a balanced unified retriever config
            unified_config = create_example_unified_config("balanced")
            retriever_configs["unified"] = {
                "mode": "unified",
                "unified_config": unified_config,
                "k": 10
            }
            logger.info(f"Unified config created: {list(unified_config.keys())}")
    
    # Create retrievers
    factory = HybridRetrieverFactory(
        vectorstore=vector_store_manager.vector_store,
        llm=llm,
        embeddings=vector_store_manager.embeddings
    )
    retrievers = {}
    
    for name, config in retriever_configs.items():
        try:
            # Handle unified retriever specially
            if config.get("mode") == "unified":
                logger.info(f"Creating unified retriever: {name}")
                retriever_config = RetrieverConfig(
                    mode=RetrieverMode.UNIFIED,
                    k=config.get("k", 10),
                    unified_config=config.get("unified_config", {})
                )
                retriever = factory.create_retriever(retriever_config)
                if retriever:
                    logger.info(f"Successfully created unified retriever: {name}")
                else:
                    logger.error(f"Failed to create unified retriever: {name}")
            else:
                retriever = factory.create_retriever(config)
                
            if retriever:
                retrievers[name] = retriever
        except Exception as e:
            import traceback
            logger.error(f"Failed to create retriever {name}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    # Define weights based on retriever importance
    # Note: multi_query removed - adds 6s latency for 0.1 weight, not worth the cost
    default_weights = {
        "vector_similarity": 0.4,
        "vector_mmr": 0.2,
        "bm25": 0.3,
        "unified": 0.5,
    }
    if retrievers:
        fallback_weight = 1.0 / len(retrievers)
    else:
        fallback_weight = 1.0
    weights = {name: default_weights.get(name, fallback_weight) for name in retrievers.keys()}
    
    reranker = None
    if enable_reranker:
        # Detect best available device
        device = "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
                logger.info("Using CUDA for reranker")
            elif torch.backends.mps.is_available():
                device = "mps"
                logger.info("Using MPS for reranker")
            else:
                logger.info("Using CPU for reranker")
        except ImportError:
            logger.warning("Torch not available, defaulting to CPU for reranker")

        reranker = CrossEncoderReranker(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            device=device
        )
    
    # Create table ranker for table-specific queries
    table_ranker = TableRanker()

    # Create RRF merger with per-request rrf_k override if provided
    effective_rrf_k = rrf_k if rrf_k is not None else getattr(settings, "rrf_k", 60)
    if rrf_k is not None:
        logger.info(f"Using per-request RRF k={effective_rrf_k}")
    rrf_merger = RRFMerger(
        k=effective_rrf_k,
        normalize_scores=getattr(settings, "rrf_normalize_scores", True),
        score_threshold=getattr(settings, "rrf_score_threshold", 0.0)
    )

    pipeline = ParallelRetrievalPipeline(
        retrievers=retrievers,
        weights=weights,
        concurrency_limit=settings.parallel_retrieval_limit,
        timeout_per_retriever=settings.retriever_timeout,
        reranker=reranker,
        table_ranker=table_ranker,
        rrf_merger=rrf_merger
    )
    
    # Wrap with stateful pipeline if requested
    if enable_stateful is None:
        enable_stateful = settings.enable_stateful_retrieval
        
    if enable_stateful:
        from app.pipelines.stateful_retrieval import StatefulRetrievalPipeline
        from app.pipelines.query_optimizer import QueryOptimizer
        
        logger.info("Wrapping pipeline with stateful retrieval (LangGraph + Redis)")
        query_optimizer = QueryOptimizer(llm=llm)
        
        return StatefulRetrievalPipeline(
            parallel_pipeline=pipeline,
            query_optimizer=query_optimizer,
            redis_url=settings.redis_url,
            max_iterations=settings.max_retrieval_iterations,
            relevance_threshold=settings.relevance_threshold,
            enable_checkpointing=bool(settings.redis_url)
        )
    
    return pipeline
