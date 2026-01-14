"""Semantic Caching for Query Responses

This module implements a semantic similarity-based cache that stores query embeddings
with responses to efficiently retrieve cached results for semantically similar queries.
"""

import json
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime, timedelta
import hashlib
import redis
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)


class SemanticCache:
    """Semantic similarity-based cache for query responses"""
    
    def __init__(
        self,
        embeddings: Embeddings,
        redis_client: Optional[redis.Redis] = None,
        similarity_threshold: float = 0.95,
        ttl: int = 3600,  # 1 hour default TTL
        max_cache_size: int = 10000,
        namespace: str = "semantic_cache"
    ):
        """Initialize semantic cache
        
        Args:
            embeddings: Embeddings model for encoding queries
            redis_client: Redis client for persistent storage
            similarity_threshold: Minimum similarity for cache hit
            ttl: Time-to-live in seconds for cached entries
            max_cache_size: Maximum number of cached entries
            namespace: Redis key namespace
        """
        self.embeddings = embeddings
        self.redis_client = redis_client
        self.similarity_threshold = similarity_threshold
        self.ttl = ttl
        self.max_cache_size = max_cache_size
        self.namespace = namespace
        
        # In-memory cache for embeddings (to avoid re-computing)
        self.embedding_cache: Dict[str, np.ndarray] = {}
        
        # Common query patterns for cache warming
        self.common_patterns = [
            "What are the travel claim procedures?",
            "How do I submit a travel claim?",
            "What is per diem allowance?",
            "What are the meal rates?",
            "How to book government travel?",
            "What documents do I need for travel?",
            "Travel advance procedures",
            "International travel requirements",
            "Leave travel assistance",
            "Relocation benefits"
        ]
    
    def _get_cache_key(self, query_hash: str) -> str:
        """Generate Redis key for query"""
        return f"{self.namespace}:{query_hash}"
    
    def _compute_embedding(self, text: str) -> np.ndarray:
        """Compute embedding for text with caching"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        if text_hash in self.embedding_cache:
            return self.embedding_cache[text_hash]
        
        embedding = np.array(self.embeddings.embed_query(text))
        self.embedding_cache[text_hash] = embedding
        
        # Limit in-memory cache size
        if len(self.embedding_cache) > self.max_cache_size:
            # Remove oldest entries (simple FIFO)
            keys_to_remove = list(self.embedding_cache.keys())[:100]
            for key in keys_to_remove:
                del self.embedding_cache[key]
        
        return embedding
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    async def get(
        self, 
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Tuple[List[Document], Dict[str, Any]]]:
        """Retrieve cached response for semantically similar query
        
        Args:
            query: The query to lookup
            context: Optional context for cache key generation
            
        Returns:
            Tuple of (documents, metadata) if cache hit, None otherwise
        """
        if not self.redis_client:
            return None
        
        try:
            # Compute query embedding
            query_embedding = self._compute_embedding(query)
            
            # Get all cached queries
            pattern = f"{self.namespace}:*"
            cache_keys = self.redis_client.keys(pattern)
            
            best_match = None
            best_similarity = 0.0
            
            for key in cache_keys:
                try:
                    # Get cached entry
                    cached_data = self.redis_client.get(key)
                    if not cached_data:
                        continue
                    
                    entry = json.loads(cached_data)
                    
                    # Check if entry is expired
                    if entry.get("expires_at", 0) < time.time():
                        self.redis_client.delete(key)
                        continue
                    
                    # Compare embeddings
                    cached_embedding = np.array(entry["embedding"])
                    similarity = self._cosine_similarity(query_embedding, cached_embedding)
                    
                    if similarity > best_similarity and similarity >= self.similarity_threshold:
                        best_similarity = similarity
                        best_match = entry
                        
                except Exception as e:
                    logger.warning(f"Error processing cache key {key}: {e}")
                    continue
            
            if best_match:
                # Reconstruct documents
                documents = [
                    Document(
                        page_content=doc["page_content"],
                        metadata=doc["metadata"]
                    )
                    for doc in best_match["documents"]
                ]
                
                metadata = best_match.get("metadata", {})
                metadata["cache_hit"] = True
                metadata["similarity_score"] = best_similarity
                metadata["cached_query"] = best_match["query"]
                
                logger.info(f"Semantic cache hit for query: {query} (similarity: {best_similarity:.3f})")
                return documents, metadata
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving from semantic cache: {e}")
            return None
    
    async def set(
        self,
        query: str,
        documents: List[Document],
        metadata: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None
    ) -> bool:
        """Store query response in semantic cache
        
        Args:
            query: The query
            documents: Retrieved documents
            metadata: Optional metadata
            context: Optional context
            ttl: Optional TTL override
            
        Returns:
            Success status
        """
        if not self.redis_client:
            return False
        
        try:
            # Compute query embedding
            query_embedding = self._compute_embedding(query)
            
            # Prepare cache entry
            entry = {
                "query": query,
                "embedding": query_embedding.tolist(),
                "documents": [
                    {
                        "page_content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in documents
                ],
                "metadata": metadata or {},
                "context": context or {},
                "created_at": time.time(),
                "expires_at": time.time() + (ttl or self.ttl)
            }
            
            # Generate cache key
            query_hash = hashlib.md5(f"{query}{context}".encode()).hexdigest()
            cache_key = self._get_cache_key(query_hash)
            
            # Store in Redis
            self.redis_client.setex(
                cache_key,
                ttl or self.ttl,
                json.dumps(entry)
            )
            
            logger.info(f"Stored in semantic cache: {query}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing in semantic cache: {e}")
            return False
    
    async def warm_cache(self, retriever: Any) -> int:
        """Warm cache with common query patterns
        
        Args:
            retriever: The retriever to use for generating responses
            
        Returns:
            Number of patterns cached
        """
        cached_count = 0
        
        for pattern in self.common_patterns:
            try:
                # Check if already cached
                cached_result = await self.get(pattern)
                if cached_result:
                    continue
                
                # Retrieve documents
                documents = await retriever.ainvoke(pattern)
                
                # Cache the result
                success = await self.set(
                    pattern,
                    documents,
                    metadata={"warmed": True}
                )
                
                if success:
                    cached_count += 1
                    
            except Exception as e:
                logger.warning(f"Error warming cache for pattern '{pattern}': {e}")
        
        logger.info(f"Warmed cache with {cached_count} patterns")
        return cached_count
    
    async def invalidate(self, pattern: Optional[str] = None) -> int:
        """Invalidate cache entries
        
        Args:
            pattern: Optional pattern to match for invalidation
            
        Returns:
            Number of entries invalidated
        """
        if not self.redis_client:
            return 0
        
        try:
            if pattern:
                # Invalidate entries matching pattern
                search_pattern = f"{self.namespace}:*{pattern}*"
            else:
                # Invalidate all entries
                search_pattern = f"{self.namespace}:*"
            
            keys = self.redis_client.keys(search_pattern)
            
            if keys:
                deleted = self.redis_client.delete(*keys)
                logger.info(f"Invalidated {deleted} cache entries")
                return deleted
            
            return 0
            
        except Exception as e:
            logger.error(f"Error invalidating cache: {e}")
            return 0
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics
        
        Returns:
            Dictionary with cache stats
        """
        if not self.redis_client:
            return {"enabled": False}
        
        try:
            pattern = f"{self.namespace}:*"
            keys = self.redis_client.keys(pattern)
            
            total_size = 0
            expired_count = 0
            
            for key in keys:
                try:
                    value = self.redis_client.get(key)
                    if value:
                        total_size += len(value)
                        entry = json.loads(value)
                        if entry.get("expires_at", 0) < time.time():
                            expired_count += 1
                except:
                    pass
            
            return {
                "enabled": True,
                "total_entries": len(keys),
                "expired_entries": expired_count,
                "total_size_bytes": total_size,
                "in_memory_embeddings": len(self.embedding_cache),
                "similarity_threshold": self.similarity_threshold,
                "ttl_seconds": self.ttl
            }
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"enabled": True, "error": str(e)}
