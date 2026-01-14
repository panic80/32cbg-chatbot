"""Embedding cache service for faster document processing."""

import hashlib
import json
import pickle
from typing import List, Dict, Optional, Tuple
import numpy as np
from datetime import datetime

from app.core.logging import get_logger
from app.services.cache import CacheService

logger = get_logger(__name__)


class EmbeddingCacheService:
    """Cache service specifically for embeddings."""

    def __init__(
        self,
        cache_service: CacheService,
        ttl: int = 604800,
        metadata_keys: Optional[List[str]] = None
    ):
        """Initialize embedding cache.

        Args:
            cache_service: Redis cache service instance
            ttl: Time to live in seconds (default: 1 week)
            metadata_keys: Optional list of metadata keys to include in cache key.
                          Use when different metadata values should produce different embeddings.
                          Example: ["language", "doc_type"] if embeddings vary by language.
        """
        self.cache_service = cache_service
        self.ttl = ttl
        self.metadata_keys = metadata_keys or []
        self.cache_hits = 0
        self.cache_misses = 0

    def _generate_embedding_key(
        self,
        text: str,
        model: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """Generate a unique cache key for text, model, and optional metadata.

        Args:
            text: The text content
            model: Embedding model name
            metadata: Optional metadata dict. Only keys specified in self.metadata_keys
                     will be included in the cache key.

        Returns:
            Cache key string
        """
        # Create hash of text + model for consistent keys
        content = f"{text}:{model}"

        # Add metadata values if configured and provided
        if self.metadata_keys and metadata:
            # Sort keys for consistent ordering
            for key in sorted(self.metadata_keys):
                value = metadata.get(key)
                if value is not None:
                    content += f":{key}={value}"

        hash_obj = hashlib.sha256(content.encode('utf-8'))
        return f"embedding:{hash_obj.hexdigest()}"
        
    async def get_embedding(
        self,
        text: str,
        model: str,
        metadata: Optional[Dict] = None
    ) -> Optional[List[float]]:
        """Get cached embedding if available.

        Args:
            text: Text that was embedded
            model: Embedding model name
            metadata: Optional metadata to include in cache key lookup

        Returns:
            Embedding vector if cached, None otherwise
        """
        try:
            key = self._generate_embedding_key(text, model, metadata)
            cached_data = await self.cache_service.get(key)

            if cached_data:
                self.cache_hits += 1
                # Deserialize the numpy array
                embedding = pickle.loads(cached_data.encode('latin-1'))
                logger.debug(f"Embedding cache hit for text hash: {key}")
                return embedding.tolist()
            else:
                self.cache_misses += 1
                return None

        except Exception as e:
            logger.warning(f"Failed to get cached embedding: {e}")
            return None

    async def set_embedding(
        self,
        text: str,
        model: str,
        embedding: List[float],
        metadata: Optional[Dict] = None
    ) -> bool:
        """Cache an embedding.

        Args:
            text: Text that was embedded
            model: Embedding model name
            embedding: The embedding vector
            metadata: Optional metadata to include in cache key

        Returns:
            True if successfully cached
        """
        try:
            key = self._generate_embedding_key(text, model, metadata)
            # Convert to numpy array for efficient storage
            embedding_array = np.array(embedding, dtype=np.float32)
            # Serialize the numpy array
            serialized = pickle.dumps(embedding_array).decode('latin-1')

            success = await self.cache_service.set(key, serialized, ttl=self.ttl)
            if success:
                logger.debug(f"Cached embedding for text hash: {key}")
            return success

        except Exception as e:
            logger.warning(f"Failed to cache embedding: {e}")
            return False
            
    async def get_embeddings_batch(
        self, 
        texts: List[str], 
        model: str
    ) -> Tuple[Dict[int, List[float]], List[int]]:
        """Get multiple cached embeddings.
        
        Args:
            texts: List of texts
            model: Embedding model name
            
        Returns:
            Tuple of (cached embeddings by index, list of missing indices)
        """
        cached_embeddings = {}
        missing_indices = []
        
        for i, text in enumerate(texts):
            embedding = await self.get_embedding(text, model)
            if embedding is not None:
                cached_embeddings[i] = embedding
            else:
                missing_indices.append(i)
                
        logger.info(
            f"Embedding cache batch lookup: {len(cached_embeddings)} hits, "
            f"{len(missing_indices)} misses out of {len(texts)} total"
        )
        
        return cached_embeddings, missing_indices
        
    async def set_embeddings_batch(
        self, 
        texts: List[str], 
        model: str, 
        embeddings: List[List[float]]
    ) -> int:
        """Cache multiple embeddings.
        
        Args:
            texts: List of texts
            model: Embedding model name
            embeddings: List of embedding vectors
            
        Returns:
            Number of successfully cached embeddings
        """
        if len(texts) != len(embeddings):
            logger.error("Texts and embeddings length mismatch")
            return 0
            
        success_count = 0
        for text, embedding in zip(texts, embeddings):
            if await self.set_embedding(text, model, embedding):
                success_count += 1
                
        logger.info(f"Cached {success_count} out of {len(texts)} embeddings")
        return success_count
        
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0
        
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "total_requests": total,
            "hit_rate": round(hit_rate, 3)
        }
        
    def reset_stats(self):
        """Reset cache statistics."""
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info("Embedding cache statistics reset")