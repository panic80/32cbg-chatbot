"""Cache service for RAG system."""

import json
import hashlib
from typing import Optional, Any, Dict, Callable
import redis.asyncio as redis
from datetime import timedelta
from functools import wraps
import asyncio

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class CacheService:
    """Redis-based cache service."""
    
    def __init__(self):
        """Initialize cache service."""
        self.redis_client: Optional[redis.Redis] = None
        self.enabled = bool(settings.redis_url)
        
    async def connect(self) -> None:
        """Connect to Redis."""
        if not self.enabled:
            logger.info("Cache service disabled - no Redis URL configured")
            return
            
        try:
            self.redis_client = redis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            
            # Test connection
            await self.redis_client.ping()
            logger.info("Connected to Redis cache")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.enabled = False
            
    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Disconnected from Redis cache")
            
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not self.enabled or not self.redis_client:
            return None
            
        try:
            value = await self.redis_client.get(key)
            if value:
                return json.loads(value)
            return None
            
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            return None
            
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """Set value in cache."""
        if not self.enabled or not self.redis_client:
            return False
            
        try:
            serialized = json.dumps(value)
            
            if ttl:
                await self.redis_client.setex(
                    key,
                    timedelta(seconds=ttl),
                    serialized
                )
            else:
                await self.redis_client.set(key, serialized)
                
            return True
            
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            return False
            
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if not self.enabled or not self.redis_client:
            return False
            
        try:
            await self.redis_client.delete(key)
            return True
            
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            return False
            
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        if not self.enabled or not self.redis_client:
            return False
            
        try:
            return await self.redis_client.exists(key) > 0
            
        except Exception as e:
            logger.error(f"Cache exists error for key {key}: {e}")
            return False
            
    async def clear_all(self) -> bool:
        """Clear all cache entries."""
        if not self.enabled or not self.redis_client:
            return True
            
        try:
            await self.redis_client.flushdb()
            logger.info("Cleared all cache entries")
            return True
            
        except Exception as e:
            logger.error(f"Cache clear all error: {e}")
            return False
            
    def make_key(self, prefix: str, *args) -> str:
        """Create cache key from prefix and arguments."""
        parts = [prefix] + [str(arg) for arg in args]
        return ":".join(parts)
        
    def make_embedding_key(self, text: str) -> str:
        """Create cache key for embeddings."""
        # Hash the text to create a consistent key
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"embedding:{text_hash}"
        
    def make_query_key(self, query: str, filters: Optional[Dict] = None) -> str:
        """Create cache key for query results."""
        # Include filters in the key
        key_parts = [query]
        if filters:
            key_parts.append(json.dumps(filters, sort_keys=True))
            
        key_content = "|".join(key_parts)
        key_hash = hashlib.md5(key_content.encode()).hexdigest()
        return f"query:{key_hash}"


class EmbeddingCache:
    """Specialized cache for embeddings."""
    
    def __init__(self, cache_service: CacheService):
        """Initialize embedding cache."""
        self.cache = cache_service
        self.ttl = settings.embedding_cache_ttl
        
    async def get_embedding(self, text: str) -> Optional[list]:
        """Get cached embedding."""
        key = self.cache.make_embedding_key(text)
        return await self.cache.get(key)
        
    async def set_embedding(self, text: str, embedding: list) -> bool:
        """Cache embedding."""
        key = self.cache.make_embedding_key(text)
        return await self.cache.set(key, embedding, self.ttl)
        
    async def get_batch(self, texts: list) -> Dict[str, list]:
        """Get multiple embeddings from cache."""
        results = {}
        
        for text in texts:
            embedding = await self.get_embedding(text)
            if embedding:
                results[text] = embedding
                
        return results
        
    async def set_batch(self, embeddings: Dict[str, list]) -> int:
        """Cache multiple embeddings."""
        count = 0
        
        for text, embedding in embeddings.items():
            if await self.set_embedding(text, embedding):
                count += 1
                
        return count


class QueryCache:
    """Specialized cache for query results."""
    
    def __init__(self, cache_service: CacheService):
        """Initialize query cache."""
        self.cache = cache_service
        self.ttl = settings.cache_ttl
        
    async def get_results(
        self,
        query: str,
        filters: Optional[Dict] = None
    ) -> Optional[Dict]:
        """Get cached query results."""
        key = self.cache.make_query_key(query, filters)
        return await self.cache.get(key)
        
    async def set_results(
        self,
        query: str,
        results: Dict,
        filters: Optional[Dict] = None
    ) -> bool:
        """Cache query results."""
        key = self.cache.make_query_key(query, filters)
        return await self.cache.set(key, results, self.ttl)


# Global cache service instance
_cache_service: Optional[CacheService] = None


def get_cache_service() -> Optional[CacheService]:
    """Get the global cache service instance."""
    return _cache_service


def set_cache_service(cache: CacheService) -> None:
    """Set the global cache service instance."""
    global _cache_service
    _cache_service = cache


def cache_result(ttl: int = 3600, key_prefix: str = "func"):
    """
    Decorator for caching function results.
    
    Args:
        ttl: Time to live in seconds
        key_prefix: Prefix for cache keys
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            cache = get_cache_service()
            if not cache or not cache.enabled:
                # No cache available, just call the function
                return await func(*args, **kwargs)
            
            # Create cache key from function name and arguments
            cache_key_parts = [key_prefix, func.__name__]
            
            # Add string representations of args
            for arg in args[1:]:  # Skip 'self' if present
                if hasattr(arg, '__class__') and hasattr(arg.__class__, '__name__'):
                    # Skip complex objects
                    continue
                cache_key_parts.append(str(arg))
            
            # Add sorted kwargs
            for k, v in sorted(kwargs.items()):
                cache_key_parts.append(f"{k}={v}")
            
            cache_key = ":".join(cache_key_parts)
            
            # Try to get from cache
            cached_result = await cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {cache_key}")
                return cached_result
            
            # Call the function
            result = await func(*args, **kwargs)
            
            # Cache the result
            await cache.set(cache_key, result, ttl)
            logger.debug(f"Cached result for {cache_key}")
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, run in event loop
            loop = asyncio.get_event_loop()
            cache = get_cache_service()
            if not cache or not cache.enabled:
                return func(*args, **kwargs)
            
            # Create cache key
            cache_key_parts = [key_prefix, func.__name__]
            for arg in args[1:]:  # Skip 'self'
                if hasattr(arg, '__class__') and hasattr(arg.__class__, '__name__'):
                    continue
                cache_key_parts.append(str(arg))
            for k, v in sorted(kwargs.items()):
                cache_key_parts.append(f"{k}={v}")
            cache_key = ":".join(cache_key_parts)
            
            # Try to get from cache
            cached_result = loop.run_until_complete(cache.get(cache_key))
            if cached_result is not None:
                logger.debug(f"Cache hit for {cache_key}")
                return cached_result
            
            # Call the function
            result = func(*args, **kwargs)
            
            # Cache the result
            loop.run_until_complete(cache.set(cache_key, result, ttl))
            logger.debug(f"Cached result for {cache_key}")
            
            return result
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator