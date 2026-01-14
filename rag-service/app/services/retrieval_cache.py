"""
L2 Retrieval Cache for caching merged document results before reranking.

This cache stores merged candidate sets with RRF scores to avoid recomputing
expensive retrieval operations. Uses composite keys that include all relevant
parameters to ensure cache correctness.
"""

import json
import hashlib
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

from langchain_core.documents import Document

from app.services.cache import CacheService
from app.components.rrf_merger import RRFDocument
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievalCacheKey:
    """Components for building retrieval cache keys."""
    query_hash: str
    index_version: str
    retriever_bitmask: str  # Which retrievers were used (e.g., "dense|sparse|bm25")
    rrf_k: int
    dedup_params: str  # Hash of deduplication parameters
    max_docs: Optional[int] = None
    
    def to_cache_key(self) -> str:
        """Convert to Redis cache key."""
        parts = [
            "retrieval_l2",
            self.index_version,
            self.retriever_bitmask,
            f"k{self.rrf_k}",
            self.dedup_params,
            self.query_hash
        ]
        
        if self.max_docs is not None:
            parts.append(f"max{self.max_docs}")
            
        return ":".join(parts)


@dataclass
class CachedRetrievalResult:
    """Cached retrieval result with metadata."""
    documents: List[Dict[str, Any]]  # Serialized RRFDocument data
    query: str
    cached_at: str  # ISO timestamp
    ttl_seconds: int
    retriever_stats: Dict[str, Any]
    
    def to_rrf_documents(self) -> List[RRFDocument]:
        """Convert serialized data back to RRFDocument objects."""
        rrf_docs = []
        
        for doc_data in self.documents:
            # Reconstruct Document
            document = Document(
                page_content=doc_data['page_content'],
                metadata=doc_data['metadata']
            )
            
            # Reconstruct RRFDocument
            rrf_doc = RRFDocument(
                document=document,
                rrf_score=doc_data['rrf_score'],
                retriever_ranks=doc_data['retriever_ranks'],
                retriever_scores=doc_data['retriever_scores']
            )
            rrf_docs.append(rrf_doc)
            
        return rrf_docs


@dataclass
class RetrievalCacheStats:
    """Statistics for retrieval cache performance."""
    total_requests: int
    cache_hits: int
    cache_misses: int
    cache_invalidations: int
    avg_hit_time_ms: float
    avg_miss_time_ms: float
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests


class RetrievalL2Cache:
    """
    L2 cache for retrieval results - caches merged document sets before reranking.
    
    This cache sits between the retrieval/RRF merge phase and reranking,
    avoiding expensive document retrieval and merging operations.
    """
    
    def __init__(
        self, 
        cache_service: CacheService,
        default_ttl: int = 86400,  # 24 hours
        enable_stats: bool = True
    ):
        """
        Initialize L2 retrieval cache.
        
        Args:
            cache_service: Underlying cache service (Redis)
            default_ttl: Default time-to-live in seconds
            enable_stats: Whether to track cache statistics
        """
        self.cache = cache_service
        self.default_ttl = default_ttl
        self.enable_stats = enable_stats
        
        # Cache statistics
        self._stats = RetrievalCacheStats(
            total_requests=0,
            cache_hits=0,
            cache_misses=0,
            cache_invalidations=0,
            avg_hit_time_ms=0.0,
            avg_miss_time_ms=0.0
        )
        
        # Current index version (for cache invalidation)
        self.current_index_version = self._get_index_version()
        
    async def get(
        self,
        query: str,
        retriever_names: List[str],
        rrf_k: int,
        dedup_params: Dict[str, Any],
        max_docs: Optional[int] = None
    ) -> Optional[Tuple[List[RRFDocument], Dict[str, Any]]]:
        """
        Get cached retrieval results.
        
        Args:
            query: Search query
            retriever_names: List of retriever names used
            rrf_k: RRF k parameter
            dedup_params: Deduplication parameters
            max_docs: Maximum documents requested
            
        Returns:
            Tuple of (RRF documents, retriever stats) if cached, None otherwise
        """
        # Check if cache is enabled
        if not self.cache.enabled:
            return None
        
        start_time = datetime.now()
        
        if self.enable_stats:
            self._stats.total_requests += 1
        
        # Build cache key
        cache_key_obj = self._build_cache_key(
            query, retriever_names, rrf_k, dedup_params, max_docs
        )
        cache_key = cache_key_obj.to_cache_key()
        
        try:
            # Try to get from cache
            cached_data = await self.cache.get(cache_key)
            
            if cached_data:
                # Parse cached result
                cached_result = CachedRetrievalResult(**cached_data)
                
                # Convert back to RRFDocument objects
                rrf_documents = cached_result.to_rrf_documents()
                
                if self.enable_stats:
                    self._stats.cache_hits += 1
                    hit_time = (datetime.now() - start_time).total_seconds() * 1000
                    self._update_avg_time("hit", hit_time)
                
                logger.debug(f"Cache hit for query hash: {cache_key_obj.query_hash[:8]}...")
                return rrf_documents, cached_result.retriever_stats
            
            if self.enable_stats:
                self._stats.cache_misses += 1
                miss_time = (datetime.now() - start_time).total_seconds() * 1000
                self._update_avg_time("miss", miss_time)
            
            logger.debug(f"Cache miss for query hash: {cache_key_obj.query_hash[:8]}...")
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving from L2 cache: {e}")
            if self.enable_stats:
                self._stats.cache_misses += 1
            return None
    
    async def set(
        self,
        query: str,
        retriever_names: List[str],
        rrf_k: int,
        dedup_params: Dict[str, Any],
        rrf_documents: List[RRFDocument],
        retriever_stats: Dict[str, Any],
        max_docs: Optional[int] = None,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache retrieval results.
        
        Args:
            query: Search query
            retriever_names: List of retriever names used
            rrf_k: RRF k parameter
            dedup_params: Deduplication parameters
            rrf_documents: RRF documents to cache
            retriever_stats: Statistics from retrieval operation
            max_docs: Maximum documents requested
            ttl: Time-to-live override
            
        Returns:
            True if successfully cached, False otherwise
        """
        # Check if cache is enabled
        if not self.cache.enabled:
            return False
        
        # Build cache key
        cache_key_obj = self._build_cache_key(
            query, retriever_names, rrf_k, dedup_params, max_docs
        )
        cache_key = cache_key_obj.to_cache_key()
        
        try:
            # Serialize RRFDocument objects
            serialized_docs = []
            for rrf_doc in rrf_documents:
                doc_data = {
                    'page_content': rrf_doc.document.page_content,
                    'metadata': rrf_doc.document.metadata,
                    'rrf_score': rrf_doc.rrf_score,
                    'retriever_ranks': rrf_doc.retriever_ranks,
                    'retriever_scores': rrf_doc.retriever_scores
                }
                serialized_docs.append(doc_data)
            
            # Create cached result
            cached_result = CachedRetrievalResult(
                documents=serialized_docs,
                query=query,
                cached_at=datetime.now().isoformat(),
                ttl_seconds=ttl or self.default_ttl,
                retriever_stats=retriever_stats
            )
            
            # Cache the result
            success = await self.cache.set(
                cache_key,
                asdict(cached_result),
                ttl or self.default_ttl
            )
            
            if success:
                logger.debug(f"Cached {len(rrf_documents)} docs for query hash: {cache_key_obj.query_hash[:8]}...")
            else:
                logger.warning(f"Failed to cache results for query hash: {cache_key_obj.query_hash[:8]}...")
            
            return success
            
        except Exception as e:
            logger.error(f"Error caching L2 results: {e}")
            return False
    
    async def invalidate_by_index_version(self, old_version: Optional[str] = None) -> int:
        """
        Invalidate cache entries for a specific index version.
        
        Args:
            old_version: Index version to invalidate (None = current version)
            
        Returns:
            Number of entries invalidated
        """
        if not self.cache.enabled:
            return 0
        
        version_to_invalidate = old_version or self.current_index_version
        pattern = f"retrieval_l2:{version_to_invalidate}:*"
        
        try:
            # This would need to be implemented in the base cache service
            # For now, we'll track invalidations but not actually delete
            # In production, you'd use Redis SCAN + DEL commands
            
            if self.enable_stats:
                self._stats.cache_invalidations += 1
            
            logger.info(f"Would invalidate L2 cache entries for index version: {version_to_invalidate}")
            return 1  # Placeholder
            
        except Exception as e:
            logger.error(f"Error invalidating L2 cache: {e}")
            return 0
    
    async def clear_all(self) -> bool:
        """Clear all L2 cache entries."""
        try:
            # In a real implementation, you'd use pattern matching
            # For now, delegate to underlying cache
            result = await self.cache.clear_all()
            
            if result and self.enable_stats:
                # Reset stats
                self._stats = RetrievalCacheStats(
                    total_requests=0,
                    cache_hits=0,
                    cache_misses=0,
                    cache_invalidations=0,
                    avg_hit_time_ms=0.0,
                    avg_miss_time_ms=0.0
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Error clearing L2 cache: {e}")
            return False
    
    def get_stats(self) -> RetrievalCacheStats:
        """Get cache statistics."""
        return self._stats
    
    def _build_cache_key(
        self,
        query: str,
        retriever_names: List[str],
        rrf_k: int,
        dedup_params: Dict[str, Any],
        max_docs: Optional[int]
    ) -> RetrievalCacheKey:
        """Build cache key components."""
        # Hash the query
        query_hash = hashlib.md5(query.encode('utf-8')).hexdigest()
        
        # Build retriever bitmask (sorted for consistency)
        retriever_bitmask = "|".join(sorted(retriever_names))
        
        # Hash deduplication parameters
        dedup_str = json.dumps(dedup_params, sort_keys=True)
        dedup_hash = hashlib.md5(dedup_str.encode('utf-8')).hexdigest()[:8]
        
        return RetrievalCacheKey(
            query_hash=query_hash,
            index_version=self.current_index_version,
            retriever_bitmask=retriever_bitmask,
            rrf_k=rrf_k,
            dedup_params=dedup_hash,
            max_docs=max_docs
        )
    
    def _get_index_version(self) -> str:
        """
        Get current index version for cache invalidation.
        
        In production, this would query the vector database for its version.
        For now, return a placeholder version.
        """
        # TODO: Implement actual index version retrieval
        # This could be from ChromaDB metadata, a version file, etc.
        return "v1.0.0"
    
    def _update_avg_time(self, operation: str, time_ms: float) -> None:
        """Update average time statistics."""
        if operation == "hit":
            if self._stats.cache_hits == 1:
                self._stats.avg_hit_time_ms = time_ms
            else:
                # Running average
                n = self._stats.cache_hits
                self._stats.avg_hit_time_ms = (self._stats.avg_hit_time_ms * (n-1) + time_ms) / n
        
        elif operation == "miss":
            if self._stats.cache_misses == 1:
                self._stats.avg_miss_time_ms = time_ms
            else:
                # Running average  
                n = self._stats.cache_misses
                self._stats.avg_miss_time_ms = (self._stats.avg_miss_time_ms * (n-1) + time_ms) / n


def create_retrieval_l2_cache(
    cache_service: CacheService,
    ttl: int = 86400,
    enable_stats: bool = True
) -> RetrievalL2Cache:
    """
    Factory function to create L2 retrieval cache.
    
    Args:
        cache_service: Underlying cache service
        ttl: Default time-to-live in seconds (24 hours)
        enable_stats: Whether to track cache statistics
        
    Returns:
        Configured RetrievalL2Cache instance
    """
    return RetrievalL2Cache(
        cache_service=cache_service,
        default_ttl=ttl,
        enable_stats=enable_stats
    )