"""Unified cache system for RAG service.

This package provides a consolidated caching system that replaces
multiple competing cache implementations with a single, layered
architecture.

Layers:
- L1: Embeddings (7 day TTL) - Query and document embeddings
- L2: Retrieval (24 hour TTL) - Retrieved document sets
- L3: Responses (6 hour TTL) - LLM generated responses

Usage:
    from app.services.cache import LayeredCacheService, CacheKeyGenerator

    cache = LayeredCacheService(backend)

    # Embeddings
    embedding = await cache.get_embedding("query text")
    await cache.set_embedding("query text", embedding_vector)

    # Retrieval results
    docs = await cache.get_retrieval_results("query", retrievers, rrf_k)

    # LLM responses
    response = await cache.get_response("query", context_hash, "gpt-4")
"""

from app.services.unified_cache.key_generator import CacheKeyGenerator
from app.services.unified_cache.layered_cache import LayeredCacheService, CacheConfig

__all__ = [
    "CacheKeyGenerator",
    "LayeredCacheService",
    "CacheConfig",
]
