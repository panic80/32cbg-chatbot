"""Unified cache key generation for all cache layers.

This module provides a single source of truth for cache key generation,
replacing the 4 different implementations that existed across:
- cache.py (make_key, make_embedding_key, make_query_key)
- advanced_cache.py (_make_embedding_key, _make_document_key, _make_response_key)
- embedding_cache.py (_generate_embedding_key)
- retrieval_cache.py (RetrievalCacheKey.to_cache_key)
"""

import hashlib
import json
from typing import Optional, Dict, Any, List


class CacheKeyGenerator:
    """Unified cache key generation for all cache layers.

    All keys include a version prefix to enable cache invalidation
    when the key format or cached data structure changes.

    Key format: {layer}:{version}:{...params}:{content_hash}

    Examples:
        emb:v1:text-embedding-3-small:abc123def456
        ret:v1:dense|sparse:k60:hash123
        resp:v1:gpt-4:query_hash:context_hash
    """

    VERSION = "v1"

    @classmethod
    def embedding(cls, text: str, model: str = "default") -> str:
        """Generate cache key for embeddings (L1 layer).

        Args:
            text: The text being embedded.
            model: The embedding model name/identifier.

        Returns:
            Cache key string.
        """
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        return f"emb:{cls.VERSION}:{model}:{text_hash}"

    @classmethod
    def query(cls, query: str, filters: Optional[Dict[str, Any]] = None) -> str:
        """Generate cache key for query results.

        Args:
            query: The search query.
            filters: Optional metadata filters.

        Returns:
            Cache key string.
        """
        key_parts = [query]
        if filters:
            # Sort keys for consistent ordering
            key_parts.append(json.dumps(filters, sort_keys=True))

        content = "|".join(key_parts)
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        return f"qry:{cls.VERSION}:{content_hash}"

    @classmethod
    def retrieval(
        cls,
        query: str,
        retriever_names: List[str],
        rrf_k: int,
        index_version: str = "current",
        max_docs: Optional[int] = None
    ) -> str:
        """Generate cache key for retrieval results (L2 layer).

        Args:
            query: The search query.
            retriever_names: List of retriever names used (e.g., ['dense', 'sparse']).
            rrf_k: The RRF k parameter.
            index_version: Version identifier for the index (for invalidation).
            max_docs: Optional max documents limit.

        Returns:
            Cache key string.
        """
        query_hash = hashlib.md5(query.encode('utf-8')).hexdigest()
        retriever_str = "|".join(sorted(retriever_names))

        parts = [
            f"ret:{cls.VERSION}",
            index_version,
            retriever_str,
            f"k{rrf_k}",
            query_hash
        ]

        if max_docs is not None:
            parts.append(f"max{max_docs}")

        return ":".join(parts)

    @classmethod
    def document(
        cls,
        query: str,
        retriever_type: str = "default",
        filters: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate cache key for document results.

        Args:
            query: The search query.
            retriever_type: Type of retriever used.
            filters: Optional metadata filters.

        Returns:
            Cache key string.
        """
        key_parts = [query, retriever_type]
        if filters:
            key_parts.append(json.dumps(filters, sort_keys=True))

        content = "|".join(key_parts)
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        return f"doc:{cls.VERSION}:{retriever_type}:{content_hash}"

    @classmethod
    def response(
        cls,
        query: str,
        context_hash: str,
        model: str,
        additional_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate cache key for LLM responses (L3 layer).

        Args:
            query: The user query.
            context_hash: Hash of the context provided to the LLM.
            model: The LLM model name.
            additional_params: Optional additional parameters (temperature, etc.).

        Returns:
            Cache key string.
        """
        query_hash = hashlib.md5(query.encode('utf-8')).hexdigest()

        parts = [
            f"resp:{cls.VERSION}",
            model,
            query_hash,
            context_hash
        ]

        if additional_params:
            params_str = json.dumps(additional_params, sort_keys=True)
            params_hash = hashlib.md5(params_str.encode('utf-8')).hexdigest()[:8]
            parts.append(params_hash)

        return ":".join(parts)

    @classmethod
    def classification(cls, query: str) -> str:
        """Generate cache key for query classification results.

        Args:
            query: The query being classified.

        Returns:
            Cache key string.
        """
        query_hash = hashlib.md5(query.encode('utf-8')).hexdigest()
        return f"cls:{cls.VERSION}:{query_hash}"

    @classmethod
    def hyde(cls, query: str) -> str:
        """Generate cache key for HyDE hypothetical documents.

        Args:
            query: The query for which a hypothesis was generated.

        Returns:
            Cache key string.
        """
        query_hash = hashlib.md5(query.lower().strip().encode('utf-8')).hexdigest()
        return f"hyde:{cls.VERSION}:{query_hash}"

    @classmethod
    def generic(cls, prefix: str, *args: Any) -> str:
        """Generate a generic cache key with prefix and arguments.

        Args:
            prefix: The key prefix.
            *args: Additional arguments to include in the key.

        Returns:
            Cache key string.
        """
        parts = [prefix, cls.VERSION] + [str(arg) for arg in args]
        return ":".join(parts)

    @classmethod
    def hash_content(cls, content: str) -> str:
        """Generate a hash for arbitrary content.

        Useful for creating context hashes or deduplication keys.

        Args:
            content: The content to hash.

        Returns:
            MD5 hash string.
        """
        return hashlib.md5(content.encode('utf-8')).hexdigest()
