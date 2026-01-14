"""Unified layered cache service for RAG system.

This service consolidates the 4 existing cache implementations into a
single, coherent caching layer with:
- L1: Embeddings (7 day TTL)
- L2: Retrieval results (24 hour TTL)
- L3: LLM responses (6 hour TTL)
"""

from dataclasses import dataclass
from typing import Optional, Any, List, Dict, Tuple

from app.services.unified_cache.key_generator import CacheKeyGenerator
from app.services.unified_cache.backends.base import ICacheBackend, CacheStats
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CacheConfig:
    """Configuration for cache TTLs and behavior."""

    # L1: Embeddings - long TTL (embeddings don't change)
    embedding_ttl: int = 604800  # 7 days

    # L2: Retrieval results - medium TTL
    retrieval_ttl: int = 86400  # 24 hours
    document_ttl: int = 86400  # 24 hours

    # L3: LLM responses - shorter TTL for freshness
    response_ttl: int = 21600  # 6 hours

    # Query classification - medium TTL
    classification_ttl: int = 3600  # 1 hour

    # HyDE hypothesis - long TTL (hypotheses don't change for same query)
    hyde_ttl: int = 86400  # 24 hours

    # Generic cache - default TTL
    default_ttl: int = 3600  # 1 hour


@dataclass
class LayeredCacheStats:
    """Statistics for all cache layers."""

    l1_embeddings: CacheStats
    l2_retrieval: CacheStats
    l3_responses: CacheStats

    @property
    def total_hits(self) -> int:
        """Total hits across all layers."""
        return (
            self.l1_embeddings.hits +
            self.l2_retrieval.hits +
            self.l3_responses.hits
        )

    @property
    def total_misses(self) -> int:
        """Total misses across all layers."""
        return (
            self.l1_embeddings.misses +
            self.l2_retrieval.misses +
            self.l3_responses.misses
        )

    @property
    def overall_hit_rate(self) -> float:
        """Overall hit rate across all layers."""
        total = self.total_hits + self.total_misses
        if total == 0:
            return 0.0
        return self.total_hits / total


class LayeredCacheService:
    """Unified layered cache service.

    Provides a single interface for all caching needs in the RAG system,
    with appropriate TTLs and key generation for each layer.

    Usage:
        backend = RedisBackend()
        await backend.connect()

        cache = LayeredCacheService(backend)

        # L1: Embeddings
        embedding = await cache.get_embedding("query text")
        await cache.set_embedding("query text", [0.1, 0.2, ...])

        # L2: Retrieval
        docs = await cache.get_retrieval_results("query", ["dense", "sparse"], 60)

        # L3: Responses
        response = await cache.get_response("query", "context_hash", "gpt-4")
    """

    def __init__(
        self,
        backend: ICacheBackend,
        config: Optional[CacheConfig] = None
    ):
        """Initialize layered cache service.

        Args:
            backend: The cache backend to use (Redis, Memory, etc.)
            config: Optional cache configuration for TTLs.
        """
        self.backend = backend
        self.config = config or CacheConfig()
        self.key_generator = CacheKeyGenerator

        # Per-layer statistics
        self._l1_stats = CacheStats()
        self._l2_stats = CacheStats()
        self._l3_stats = CacheStats()

    @property
    def enabled(self) -> bool:
        """Check if caching is enabled."""
        return self.backend.enabled

    @property
    def stats(self) -> LayeredCacheStats:
        """Get statistics for all cache layers."""
        return LayeredCacheStats(
            l1_embeddings=self._l1_stats,
            l2_retrieval=self._l2_stats,
            l3_responses=self._l3_stats
        )

    # =========================================================================
    # L1: Embedding Cache
    # =========================================================================

    async def get_embedding(
        self,
        text: str,
        model: str = "default"
    ) -> Optional[List[float]]:
        """Get cached embedding.

        Args:
            text: The text that was embedded.
            model: The embedding model name.

        Returns:
            The cached embedding vector, or None if not found.
        """
        key = self.key_generator.embedding(text, model)
        result = await self.backend.get(key)

        if result is not None:
            self._l1_stats.record_hit()
        else:
            self._l1_stats.record_miss()

        return result

    async def set_embedding(
        self,
        text: str,
        embedding: List[float],
        model: str = "default"
    ) -> bool:
        """Cache an embedding.

        Args:
            text: The text that was embedded.
            embedding: The embedding vector.
            model: The embedding model name.

        Returns:
            True if cached successfully.
        """
        key = self.key_generator.embedding(text, model)
        return await self.backend.set(key, embedding, self.config.embedding_ttl)

    async def get_embeddings_batch(
        self,
        texts: List[str],
        model: str = "default"
    ) -> Dict[str, List[float]]:
        """Get multiple embeddings from cache using batch operation.

        This uses pipeline/MGET for efficiency instead of N+1 calls.

        Args:
            texts: List of texts to get embeddings for.
            model: The embedding model name.

        Returns:
            Dictionary mapping text to embedding (missing texts omitted).
        """
        if not texts:
            return {}

        # Generate keys
        keys = [self.key_generator.embedding(text, model) for text in texts]
        key_to_text = dict(zip(keys, texts))

        # Batch get
        results = await self.backend.get_many(keys)

        # Map back to text keys and update stats
        text_results = {}
        for key, text in key_to_text.items():
            if key in results:
                text_results[text] = results[key]
                self._l1_stats.record_hit()
            else:
                self._l1_stats.record_miss()

        return text_results

    async def set_embeddings_batch(
        self,
        embeddings: Dict[str, List[float]],
        model: str = "default"
    ) -> int:
        """Cache multiple embeddings using batch operation.

        Args:
            embeddings: Dictionary mapping text to embedding vector.
            model: The embedding model name.

        Returns:
            Number of embeddings cached.
        """
        if not embeddings:
            return 0

        items = {
            self.key_generator.embedding(text, model): embedding
            for text, embedding in embeddings.items()
        }

        return await self.backend.set_many(items, self.config.embedding_ttl)

    # =========================================================================
    # L2: Retrieval Cache
    # =========================================================================

    async def get_retrieval_results(
        self,
        query: str,
        retriever_names: List[str],
        rrf_k: int,
        index_version: str = "current",
        max_docs: Optional[int] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """Get cached retrieval results.

        Args:
            query: The search query.
            retriever_names: List of retriever names used.
            rrf_k: The RRF k parameter.
            index_version: Version of the index (for invalidation).
            max_docs: Optional max documents limit.

        Returns:
            Cached retrieval results, or None if not found.
        """
        key = self.key_generator.retrieval(
            query, retriever_names, rrf_k, index_version, max_docs
        )
        result = await self.backend.get(key)

        if result is not None:
            self._l2_stats.record_hit()
        else:
            self._l2_stats.record_miss()

        return result

    async def set_retrieval_results(
        self,
        query: str,
        retriever_names: List[str],
        rrf_k: int,
        results: List[Dict[str, Any]],
        index_version: str = "current",
        max_docs: Optional[int] = None
    ) -> bool:
        """Cache retrieval results.

        Args:
            query: The search query.
            retriever_names: List of retriever names used.
            rrf_k: The RRF k parameter.
            results: The retrieval results to cache.
            index_version: Version of the index.
            max_docs: Optional max documents limit.

        Returns:
            True if cached successfully.
        """
        key = self.key_generator.retrieval(
            query, retriever_names, rrf_k, index_version, max_docs
        )
        return await self.backend.set(key, results, self.config.retrieval_ttl)

    async def get_documents(
        self,
        query: str,
        retriever_type: str = "default",
        filters: Optional[Dict[str, Any]] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """Get cached documents.

        Args:
            query: The search query.
            retriever_type: Type of retriever used.
            filters: Optional metadata filters.

        Returns:
            Cached documents, or None if not found.
        """
        key = self.key_generator.document(query, retriever_type, filters)
        result = await self.backend.get(key)

        if result is not None:
            self._l2_stats.record_hit()
        else:
            self._l2_stats.record_miss()

        return result

    async def set_documents(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        retriever_type: str = "default",
        filters: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Cache documents.

        Args:
            query: The search query.
            documents: The documents to cache.
            retriever_type: Type of retriever used.
            filters: Optional metadata filters.

        Returns:
            True if cached successfully.
        """
        key = self.key_generator.document(query, retriever_type, filters)
        return await self.backend.set(key, documents, self.config.document_ttl)

    # =========================================================================
    # L3: Response Cache
    # =========================================================================

    async def get_response(
        self,
        query: str,
        context_hash: str,
        model: str,
        additional_params: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Get cached LLM response.

        Args:
            query: The user query.
            context_hash: Hash of the context provided to the LLM.
            model: The LLM model name.
            additional_params: Optional additional parameters.

        Returns:
            Cached response, or None if not found.
        """
        key = self.key_generator.response(query, context_hash, model, additional_params)
        result = await self.backend.get(key)

        if result is not None:
            self._l3_stats.record_hit()
        else:
            self._l3_stats.record_miss()

        return result

    async def set_response(
        self,
        query: str,
        context_hash: str,
        model: str,
        response: Dict[str, Any],
        additional_params: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Cache an LLM response.

        Args:
            query: The user query.
            context_hash: Hash of the context provided to the LLM.
            model: The LLM model name.
            response: The response to cache.
            additional_params: Optional additional parameters.

        Returns:
            True if cached successfully.
        """
        key = self.key_generator.response(query, context_hash, model, additional_params)
        return await self.backend.set(key, response, self.config.response_ttl)

    # =========================================================================
    # Query Classification Cache
    # =========================================================================

    async def get_classification(
        self,
        query: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached query classification.

        Args:
            query: The query that was classified.

        Returns:
            Cached classification, or None if not found.
        """
        key = self.key_generator.classification(query)
        return await self.backend.get(key)

    async def set_classification(
        self,
        query: str,
        classification: Dict[str, Any]
    ) -> bool:
        """Cache a query classification.

        Args:
            query: The query that was classified.
            classification: The classification result.

        Returns:
            True if cached successfully.
        """
        key = self.key_generator.classification(query)
        return await self.backend.set(
            key, classification, self.config.classification_ttl
        )

    # =========================================================================
    # HyDE (Hypothetical Document Embeddings) Cache
    # =========================================================================

    async def get_hyde_hypothesis(self, query: str) -> Optional[str]:
        """Get cached HyDE hypothetical document.

        Args:
            query: The query for which a hypothesis was generated.

        Returns:
            Cached hypothesis string, or None if not found.
        """
        key = self.key_generator.hyde(query)
        return await self.backend.get(key)

    async def set_hyde_hypothesis(self, query: str, hypothesis: str) -> bool:
        """Cache a HyDE hypothetical document.

        Args:
            query: The query for which hypothesis was generated.
            hypothesis: The hypothetical document/answer.

        Returns:
            True if cached successfully.
        """
        key = self.key_generator.hyde(query)
        return await self.backend.set(key, hypothesis, self.config.hyde_ttl)

    # =========================================================================
    # Generic Operations
    # =========================================================================

    async def get(self, key: str) -> Optional[Any]:
        """Get a value by raw key (for custom caching)."""
        return await self.backend.get(key)

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """Set a value by raw key (for custom caching)."""
        return await self.backend.set(
            key, value, ttl or self.config.default_ttl
        )

    async def delete(self, key: str) -> bool:
        """Delete a key."""
        return await self.backend.delete(key)

    async def clear_all(self) -> bool:
        """Clear all cache entries."""
        result = await self.backend.clear_all()
        if result:
            # Reset stats
            self._l1_stats.reset()
            self._l2_stats.reset()
            self._l3_stats.reset()
        return result

    async def clear_layer(self, layer: str) -> int:
        """Clear a specific cache layer by prefix.

        Args:
            layer: The layer to clear ('l1', 'l2', 'l3', 'embedding', 'retrieval', 'response')

        Returns:
            Number of keys deleted (0 if not supported by backend).
        """
        # This would require SCAN and DELETE operations
        # For now, log a warning that this is not implemented
        logger.warning(f"clear_layer({layer}) not fully implemented - use clear_all() or manual cleanup")
        return 0

    def reset_stats(self) -> None:
        """Reset all statistics."""
        self._l1_stats.reset()
        self._l2_stats.reset()
        self._l3_stats.reset()
