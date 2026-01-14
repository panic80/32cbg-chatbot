"""Retrieval execution service for document search."""

import asyncio
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import settings
from app.core.logging import get_logger
from app.models.query import ChatRequest, RetrievalConfig
from app.pipelines.parallel_retrieval import create_parallel_pipeline
from app.services.chat.query_processor import should_use_hybrid

logger = get_logger(__name__)


class RetrievalExecutor:
    """Executes retrieval operations."""

    def __init__(
        self,
        vector_store_manager: Any,
        app_state: Any,
        llm_wrapper: Optional[Any] = None,
    ):
        """Initialize retrieval executor.

        Args:
            vector_store_manager: The vector store manager.
            app_state: Application state for cache access.
            llm_wrapper: Optional LLM for multi-query retrieval.
        """
        self.vector_store_manager = vector_store_manager
        self.app_state = app_state
        self.llm_wrapper = llm_wrapper

    async def create_pipeline(
        self,
        chat_request: ChatRequest,
        is_fast_mode: bool = False,
    ) -> Any:
        """Create or retrieve cached retrieval pipeline.

        Args:
            chat_request: The chat request.
            is_fast_mode: Whether using fast/streamlined mode (skip reranker, limited context).

        Returns:
            The retrieval pipeline.
        """
        enable_unified = getattr(settings, "enable_unified_retrieval", False)
        retriever_configs = None

        cache_service = getattr(self.app_state, "cache_service", None)
        redis_client = (
            getattr(cache_service, "redis_client", None)
            if cache_service and hasattr(cache_service, "redis_client")
            else None
        )
        enable_stateful = getattr(settings, "enable_stateful_retrieval", False)

        # Extract per-request retrieval config overrides
        retrieval_config = chat_request.retrieval_config
        has_config_overrides = retrieval_config is not None

        # Determine retrieval_k (default from settings if not overridden)
        retrieval_k = getattr(settings, "retrieval_k", 10)
        if retrieval_config and retrieval_config.retrieval_k is not None:
            retrieval_k = retrieval_config.retrieval_k
            logger.info(f"Using per-request retrieval_k: {retrieval_k}")

        # Determine enable_reranker
        enable_reranker = not is_fast_mode
        if retrieval_config and retrieval_config.enable_reranker is not None:
            enable_reranker = retrieval_config.enable_reranker
            logger.info(f"Using per-request enable_reranker: {enable_reranker}")

        if should_use_hybrid(chat_request):
            if settings.enable_bm25:
                logger.info("Hybrid search enabled - configuring BM25 + Vector retrievers")
                retriever_configs = {
                    "vector_similarity": {
                        "type": "vector",
                        "search_type": "similarity",
                        "k": retrieval_k,
                    },
                    "bm25": {
                        "type": "bm25",
                        "k": retrieval_k,
                    },
                }
            else:
                logger.info("Hybrid search requested but BM25 disabled; using vector only")
                retriever_configs = {
                    "vector_similarity": {
                        "type": "vector",
                        "search_type": "similarity",
                        "k": retrieval_k,
                    },
                }
            # Multi-query retriever disabled for performance (saves ~6s)
            # Analysis showed 0.1 RRF weight with high overlap, not worth 6s latency
            # if self.llm_wrapper:
            #     retriever_configs["multi_query"] = {
            #         "type": "multi_query",
            #         "base_retriever": "vector_similarity",
            #         "llm": self.llm_wrapper,
            #     }

        if retriever_configs is None and is_fast_mode:
            logger.info(
                "Fast mode detected - using streamlined vector retriever configuration"
            )
            retriever_configs = {
                "vector_similarity": {
                    "type": "vector",
                    "search_type": "similarity",
                    "k": max(8, getattr(settings, "smart_mode_max_chunks", 4) * 2),
                }
            }

        pipeline_cache_store = getattr(self.app_state, "retrieval_pipeline_cache", None)

        provider_key = str(chat_request.provider)
        model_key = chat_request.model or "default"
        hybrid_key = "hybrid" if should_use_hybrid(chat_request) else "vector"

        # If we have config overrides, create a unique cache key or skip cache entirely
        if has_config_overrides:
            # Build config-specific cache key components
            config_parts = []
            if retrieval_config.retrieval_k is not None:
                config_parts.append(f"k={retrieval_config.retrieval_k}")
            if retrieval_config.rrf_k is not None:
                config_parts.append(f"rrf={retrieval_config.rrf_k}")
            if retrieval_config.enable_hyde is not None:
                config_parts.append(f"hyde={retrieval_config.enable_hyde}")
            if retrieval_config.enable_reranker is not None:
                config_parts.append(f"rerank={retrieval_config.enable_reranker}")
            if retrieval_config.unified_retrieval_mode is not None:
                config_parts.append(f"mode={retrieval_config.unified_retrieval_mode}")
            config_suffix = "|".join(config_parts) if config_parts else "default"
            cache_key = f"{hybrid_key}|unified={enable_unified}|{provider_key}|{model_key}|{config_suffix}"
        else:
            cache_key = f"{hybrid_key}|unified={enable_unified}|{provider_key}|{model_key}"

        if pipeline_cache_store is not None and cache_key in pipeline_cache_store and not has_config_overrides:
            pipeline = pipeline_cache_store[cache_key]
            logger.info("Using cached retrieval pipeline: %s", cache_key)
        else:
            # Extract rrf_k override for pipeline creation
            rrf_k_override = None
            if retrieval_config and retrieval_config.rrf_k is not None:
                rrf_k_override = retrieval_config.rrf_k
                logger.info(f"Using per-request rrf_k: {rrf_k_override}")

            pipeline = await asyncio.to_thread(
                create_parallel_pipeline,
                vector_store_manager=self.vector_store_manager,
                llm=self.llm_wrapper,
                enable_unified=enable_unified,
                retriever_configs=retriever_configs,
                enable_reranker=enable_reranker,
                enable_stateful=enable_stateful,
                redis_client=redis_client,
                rrf_k=rrf_k_override,
            )
            if pipeline_cache_store is not None and not has_config_overrides:
                pipeline_cache_store[cache_key] = pipeline
                logger.info("Cached retrieval pipeline: %s", cache_key)
            elif has_config_overrides:
                logger.info("Created pipeline with config overrides (not cached): %s", cache_key)

        return pipeline

    async def retrieve(
        self,
        pipeline: Any,
        query: str,
        k: Optional[int] = None,
        is_fast_mode: bool = False,
        hyde_hypothesis: Optional[str] = None,
        hyde_generator: Optional[Any] = None,
        classification: Optional[Dict[str, Any]] = None,
        auxiliary_model: Optional[str] = None,
        auxiliary_provider: Optional[Any] = None,
    ) -> List[Tuple]:
        """Execute retrieval.

        Args:
            pipeline: The retrieval pipeline.
            query: The query string.
            k: Number of results (defaults to settings).
            is_fast_mode: Whether to apply fast mode limits (reduced chunks).
            hyde_hypothesis: Optional HyDE hypothesis for improved retrieval.
            hyde_generator: Optional HyDE generator instance for concurrent generation.
            classification: Optional query classification.
            auxiliary_model: Optional model for auxiliary tasks (HyDE).
            auxiliary_provider: Optional provider for auxiliary tasks.

        Returns:
            List of (document, score) tuples.
        """
        if k is None:
            k = getattr(settings, "max_chunks_per_query", 6)

        # Pass HyDE hypothesis string OR generator to pipeline
        if settings.enable_hyde and (hyde_hypothesis or hyde_generator):
            logger.debug("Executing retrieval with HyDE support")
            results = await pipeline.retrieve(
                query=query, 
                k=k, 
                hyde_hypothesis=hyde_hypothesis,
                hyde_generator=hyde_generator,
                auxiliary_model=auxiliary_model,
                auxiliary_provider=auxiliary_provider
            )
        else:
            results = await pipeline.retrieve(query=query, k=k, auxiliary_model=auxiliary_model, auxiliary_provider=auxiliary_provider)

        # Apply fast mode chunk limit
        if is_fast_mode:
            smart_chunk_limit = getattr(settings, "smart_mode_max_chunks", 0)
            if smart_chunk_limit:
                results = results[:smart_chunk_limit]

        return results

    async def search_supplemental(
        self,
        query: str,
        k: int = 5,
    ) -> List[Tuple]:
        """Search for supplemental documents.

        Args:
            query: The query string.
            k: Number of results.

        Returns:
            List of (document, score) tuples.
        """
        try:
            return await self.vector_store_manager.search(query, k=k)
        except Exception as exc:
            logger.debug("Supplemental search failed for '%s': %s", query, exc)
            return []
