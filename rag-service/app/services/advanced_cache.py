"""Advanced multi-level caching for the RAG system."""

import json
import hashlib
from typing import Optional, Any, Dict, List, Tuple
from datetime import datetime, timedelta
import asyncio
from enum import Enum

import redis.asyncio as redis
import numpy as np

from app.core.config import settings
from app.core.logging import get_logger
from app.services.cache import CacheService

logger = get_logger(__name__)


class CacheLevel(Enum):
    """Cache levels for multi-level caching."""
    L1_EMBEDDINGS = "l1_embeddings"
    L2_DOCUMENTS = "l2_documents"
    L3_RESPONSES = "l3_responses"


class CacheTTL:
    """TTL configuration for different cache types."""
    # Query embeddings - long TTL as they don't change
    EMBEDDINGS = 86400 * 7  # 7 days
    
    # Retrieved documents - medium TTL
    DOCUMENTS = 86400  # 1 day
    
    # LLM responses - shorter TTL for freshness
    RESPONSES = 3600 * 6  # 6 hours
    
    # Common queries - extended TTL
    COMMON_QUERIES = 86400 * 3  # 3 days


class AdvancedCacheService:
    """Advanced multi-level cache with warming and invalidation."""
    
    def __init__(self, base_cache: Optional[CacheService] = None):
        """Initialize advanced cache service."""
        self.base_cache = base_cache or CacheService()
        self._warmup_queries: List[str] = []
        self._query_frequency: Dict[str, int] = {}
        self._cache_stats = {
            "hits": {"l1": 0, "l2": 0, "l3": 0},
            "misses": {"l1": 0, "l2": 0, "l3": 0},
            "evictions": 0
        }
        
    async def initialize(self):
        """Initialize cache service."""
        if not self.base_cache.redis_client:
            await self.base_cache.connect()
            
        # Load warmup queries
        await self._load_warmup_queries()
        
    async def _load_warmup_queries(self):
        """Load common queries for cache warming."""
        # Common travel-related queries
        self._warmup_queries = [
            "meal allowance rates",
            "kilometric rates",
            "incidental allowance",
            "hotel accommodation rates",
            "travel claim process",
            "POMV rates",
            "international travel allowances",
            "TD travel directive",
            "relocation benefits",
            "posting allowances"
        ]
        
    # L1: Embedding Cache
    async def get_embedding(
        self,
        text: str,
        embedding_model: str = "default"
    ) -> Optional[List[float]]:
        """Get cached embedding (L1 cache)."""
        key = self._make_embedding_key(text, embedding_model)
        
        try:
            value = await self.base_cache.get(key)
            if value:
                self._cache_stats["hits"]["l1"] += 1
                return value
            else:
                self._cache_stats["misses"]["l1"] += 1
                return None
        except Exception as e:
            logger.error(f"L1 cache error: {e}")
            return None
            
    async def set_embedding(
        self,
        text: str,
        embedding: List[float],
        embedding_model: str = "default"
    ) -> bool:
        """Cache embedding (L1 cache)."""
        key = self._make_embedding_key(text, embedding_model)
        return await self.base_cache.set(key, embedding, CacheTTL.EMBEDDINGS)
        
    async def get_embedding_batch(
        self,
        texts: List[str],
        embedding_model: str = "default"
    ) -> Dict[str, List[float]]:
        """Get multiple embeddings from cache."""
        results = {}
        
        # Use pipeline for efficiency
        if self.base_cache.redis_client:
            async with self.base_cache.redis_client.pipeline() as pipe:
                for text in texts:
                    key = self._make_embedding_key(text, embedding_model)
                    pipe.get(key)
                    
                values = await pipe.execute()
                
                for text, value in zip(texts, values):
                    if value:
                        results[text] = json.loads(value)
                        self._cache_stats["hits"]["l1"] += 1
                    else:
                        self._cache_stats["misses"]["l1"] += 1
                        
        return results
        
    # L2: Document Cache
    async def get_documents(
        self,
        query: str,
        filters: Optional[Dict] = None,
        retriever_type: str = "default"
    ) -> Optional[List[Dict[str, Any]]]:
        """Get cached documents (L2 cache)."""
        key = self._make_document_key(query, filters, retriever_type)
        
        try:
            value = await self.base_cache.get(key)
            if value:
                self._cache_stats["hits"]["l2"] += 1
                # Track query frequency
                await self._track_query_frequency(query)
                return value
            else:
                self._cache_stats["misses"]["l2"] += 1
                return None
        except Exception as e:
            logger.error(f"L2 cache error: {e}")
            return None
            
    async def set_documents(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        filters: Optional[Dict] = None,
        retriever_type: str = "default"
    ) -> bool:
        """Cache documents (L2 cache)."""
        key = self._make_document_key(query, filters, retriever_type)
        
        # Determine TTL based on query frequency
        ttl = await self._get_adaptive_ttl(query, CacheTTL.DOCUMENTS)
        
        return await self.base_cache.set(key, documents, ttl)
        
    # L3: Response Cache  
    async def get_response(
        self,
        query: str,
        context_hash: str,
        model: str = "default"
    ) -> Optional[Dict[str, Any]]:
        """Get cached LLM response (L3 cache)."""
        key = self._make_response_key(query, context_hash, model)
        
        try:
            value = await self.base_cache.get(key)
            if value:
                self._cache_stats["hits"]["l3"] += 1
                return value
            else:
                self._cache_stats["misses"]["l3"] += 1
                return None
        except Exception as e:
            logger.error(f"L3 cache error: {e}")
            return None
            
    async def set_response(
        self,
        query: str,
        context_hash: str,
        response: Dict[str, Any],
        model: str = "default"
    ) -> bool:
        """Cache LLM response (L3 cache)."""
        key = self._make_response_key(query, context_hash, model)
        
        # Adaptive TTL based on content type
        ttl = CacheTTL.RESPONSES
        if "policy" in query.lower() or "regulation" in query.lower():
            # Policy content changes less frequently
            ttl = CacheTTL.RESPONSES * 2
            
        return await self.base_cache.set(key, response, ttl)
        
    # Cache Warming
    async def warm_cache(self, embeddings_func=None, retrieval_func=None):
        """Warm cache with common queries."""
        logger.info("Starting cache warming...")
        
        warmed = 0
        for query in self._warmup_queries:
            try:
                # Warm embeddings if function provided
                if embeddings_func and not await self.get_embedding(query):
                    embedding = await embeddings_func(query)
                    if embedding:
                        await self.set_embedding(query, embedding)
                        warmed += 1
                        
                # Warm documents if function provided
                if retrieval_func and not await self.get_documents(query):
                    documents = await retrieval_func(query)
                    if documents:
                        await self.set_documents(query, documents)
                        warmed += 1
                        
            except Exception as e:
                logger.error(f"Error warming cache for query '{query}': {e}")
                
        logger.info(f"Cache warming completed. Warmed {warmed} entries.")
        
    # Cache Invalidation
    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern."""
        if not self.base_cache.redis_client:
            return 0
            
        try:
            # Find all matching keys
            keys = []
            async for key in self.base_cache.redis_client.scan_iter(match=pattern):
                keys.append(key)
                
            if keys:
                deleted = await self.base_cache.redis_client.delete(*keys)
                self._cache_stats["evictions"] += deleted
                return deleted
                
            return 0
            
        except Exception as e:
            logger.error(f"Error invalidating pattern {pattern}: {e}")
            return 0
            
    async def invalidate_by_source(self, source_id: str):
        """Invalidate cache entries related to a source document."""
        patterns = [
            f"*:source:{source_id}:*",
            f"l2_documents:*{source_id}*",
            f"l3_responses:*"  # Invalidate all responses when source changes
        ]
        
        total_invalidated = 0
        for pattern in patterns:
            invalidated = await self.invalidate_pattern(pattern)
            total_invalidated += invalidated
            
        logger.info(f"Invalidated {total_invalidated} cache entries for source {source_id}")
        return total_invalidated
        
    # Helper Methods
    def _make_embedding_key(self, text: str, model: str) -> str:
        """Create cache key for embeddings."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"{CacheLevel.L1_EMBEDDINGS.value}:{model}:{text_hash}"
        
    def _make_document_key(
        self,
        query: str,
        filters: Optional[Dict],
        retriever_type: str
    ) -> str:
        """Create cache key for documents."""
        key_parts = [query, retriever_type]
        if filters:
            key_parts.append(json.dumps(filters, sort_keys=True))
            
        key_content = "|".join(key_parts)
        key_hash = hashlib.md5(key_content.encode()).hexdigest()
        return f"{CacheLevel.L2_DOCUMENTS.value}:{key_hash}"
        
    def _make_response_key(self, query: str, context_hash: str, model: str) -> str:
        """Create cache key for responses."""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        return f"{CacheLevel.L3_RESPONSES.value}:{model}:{query_hash}:{context_hash}"
        
    async def _track_query_frequency(self, query: str):
        """Track query frequency for adaptive caching."""
        # Normalize query
        normalized = query.lower().strip()
        
        # Update in-memory counter
        self._query_frequency[normalized] = self._query_frequency.get(normalized, 0) + 1
        
        # Persist to Redis periodically
        if self.base_cache.redis_client:
            freq_key = f"query_freq:{hashlib.md5(normalized.encode()).hexdigest()}"
            await self.base_cache.redis_client.incr(freq_key)
            await self.base_cache.redis_client.expire(freq_key, 86400 * 30)  # 30 days
            
    async def _get_adaptive_ttl(self, query: str, base_ttl: int) -> int:
        """Get adaptive TTL based on query frequency."""
        normalized = query.lower().strip()
        frequency = self._query_frequency.get(normalized, 0)
        
        # Check Redis for historical frequency
        if self.base_cache.redis_client:
            freq_key = f"query_freq:{hashlib.md5(normalized.encode()).hexdigest()}"
            redis_freq = await self.base_cache.redis_client.get(freq_key)
            if redis_freq:
                frequency = max(frequency, int(redis_freq))
                
        # Adaptive TTL based on frequency
        if frequency > 100:
            return base_ttl * 3  # Triple TTL for very common queries
        elif frequency > 50:
            return base_ttl * 2  # Double TTL for common queries
        elif frequency > 10:
            return int(base_ttl * 1.5)  # 1.5x TTL for somewhat common
        else:
            return base_ttl  # Default TTL
            
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = dict(self._cache_stats)
        
        # Calculate hit rates
        for level in ["l1", "l2", "l3"]:
            hits = stats["hits"][level]
            misses = stats["misses"][level]
            total = hits + misses
            stats[f"{level}_hit_rate"] = hits / total if total > 0 else 0
            
        # Add frequency info
        stats["top_queries"] = sorted(
            self._query_frequency.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return stats
        
    async def clear_stats(self):
        """Clear cache statistics."""
        self._cache_stats = {
            "hits": {"l1": 0, "l2": 0, "l3": 0},
            "misses": {"l1": 0, "l2": 0, "l3": 0},
            "evictions": 0
        }
        self._query_frequency.clear()


def create_context_hash(query: str, documents: List[Any], model: str) -> str:
    """Create a hash of query, document context, and model for cache keys."""
    from langchain_core.documents import Document
    
    # Include query and model in the hash
    hash_parts = [query, model]
    
    # Sort documents by ID for consistency
    def get_doc_id(doc):
        if isinstance(doc, Document):
            return doc.metadata.get("id", "")
        elif isinstance(doc, dict):
            return doc.get("id", "")
        return ""
    
    sorted_docs = sorted(documents, key=get_doc_id)
    
    # Create hash from document IDs and first 100 chars of content
    doc_content = ""
    for doc in sorted_docs[:10]:  # Limit to first 10 docs
        if isinstance(doc, Document):
            doc_id = doc.metadata.get("id", "")
            content_preview = doc.page_content[:100] if doc.page_content else ""
        elif isinstance(doc, dict):
            doc_id = doc.get("id", "")
            content_preview = doc.get("page_content", "")[:100]
        else:
            continue
            
        doc_content += f"{doc_id}:{content_preview}"
    
    hash_parts.append(doc_content)
    
    # Combine all parts for the final hash
    hash_content = "|".join(hash_parts)
    return hashlib.md5(hash_content.encode()).hexdigest()