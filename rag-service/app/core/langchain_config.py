"""LangChain configuration with Redis caching and retry logic."""

import os
from typing import Optional, Dict, Any
from langchain_core.globals import set_llm_cache
from langchain_community.cache import RedisCache
from langchain_core.caches import InMemoryCache
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import redis
from redis.exceptions import RedisError

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class LangChainConfig:
    """Configuration for LangChain with caching and optimizations."""
    
    _initialized = False
    _redis_client: Optional[redis.Redis] = None
    
    @classmethod
    def initialize(cls) -> None:
        """Initialize LangChain configuration including caching."""
        if cls._initialized:
            return
            
        # Set up caching
        cls._setup_caching()
        
        # Set other global configurations
        # Disable verbose mode for production
        os.environ["LANGCHAIN_VERBOSE"] = "false" if not settings.debug else "true"
        
        # Enable tracing only in debug mode
        if settings.debug:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_PROJECT"] = settings.app_name
        
        cls._initialized = True
        logger.info("LangChain configuration initialized")
    
    @classmethod
    def _setup_caching(cls) -> None:
        """Set up LLM caching with Redis or fallback to in-memory."""
        cache_configured = False
        
        # Try Redis cache first
        if settings.redis_url:
            try:
                # Parse Redis URL and create client
                cls._redis_client = redis.from_url(
                    settings.redis_url,
                    decode_responses=False,  # Required for binary data
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True,
                    retry_on_error=[RedisError],
                    health_check_interval=30
                )
                
                # Test connection
                cls._redis_client.ping()
                
                # Set up Redis cache with namespace
                cache = RedisCache(
                    redis_=cls._redis_client,
                    prefix="langchain:llm:",  # Namespace for LLM cache
                    ttl=settings.cache_ttl
                )
                set_llm_cache(cache)
                cache_configured = True
                logger.info("Redis LLM cache configured successfully")
                
            except Exception as e:
                logger.warning(f"Failed to configure Redis cache: {e}")
                cls._redis_client = None
        
        # Fallback to in-memory cache
        if not cache_configured:
            set_llm_cache(InMemoryCache())
            logger.info("Using in-memory LLM cache (Redis not available)")
    
    @classmethod
    def get_redis_client(cls) -> Optional[redis.Redis]:
        """Get the Redis client if available."""
        return cls._redis_client
    
    @classmethod
    def clear_cache(cls) -> None:
        """Clear the LLM cache."""
        try:
            if cls._redis_client:
                # Clear all keys with our prefix
                keys = cls._redis_client.keys("langchain:llm:*")
                if keys:
                    cls._redis_client.delete(*keys)
                    logger.info(f"Cleared {len(keys)} cached LLM responses")
            else:
                # For in-memory cache, we need to reinitialize
                set_llm_cache(InMemoryCache())
                logger.info("Cleared in-memory LLM cache")
                
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
    
    @classmethod
    def get_cache_stats(cls) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            "type": "redis" if cls._redis_client else "in-memory",
            "connected": False,
            "keys": 0,
            "memory_usage": 0
        }
        
        try:
            if cls._redis_client:
                cls._redis_client.ping()
                stats["connected"] = True
                stats["keys"] = len(cls._redis_client.keys("langchain:llm:*"))
                
                # Get memory usage if available
                info = cls._redis_client.info("memory")
                stats["memory_usage"] = info.get("used_memory_human", "0")
                
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            
        return stats


# Additional cache decorators for custom caching
def cache_embedding(ttl: Optional[int] = None):
    """Decorator for caching embedding results."""
    from functools import wraps
    import hashlib
    import json
    
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            key_data = {"args": args, "kwargs": kwargs}
            key = f"embedding:{hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()}"
            
            redis_client = LangChainConfig.get_redis_client()
            if redis_client:
                try:
                    # Try to get from cache
                    cached = redis_client.get(key)
                    if cached:
                        logger.debug(f"Cache hit for embedding: {key}")
                        return json.loads(cached)
                except Exception as e:
                    logger.warning(f"Cache get failed: {e}")
            
            # Call original function
            result = await func(*args, **kwargs)
            
            # Cache result
            if redis_client:
                try:
                    cache_ttl = ttl or settings.embedding_cache_ttl
                    redis_client.setex(key, cache_ttl, json.dumps(result))
                    logger.debug(f"Cached embedding: {key}")
                except Exception as e:
                    logger.warning(f"Cache set failed: {e}")
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Similar logic for sync functions
            key_data = {"args": args, "kwargs": kwargs}
            key = f"embedding:{hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()}"
            
            redis_client = LangChainConfig.get_redis_client()
            if redis_client:
                try:
                    cached = redis_client.get(key)
                    if cached:
                        logger.debug(f"Cache hit for embedding: {key}")
                        return json.loads(cached)
                except Exception as e:
                    logger.warning(f"Cache get failed: {e}")
            
            result = func(*args, **kwargs)
            
            if redis_client:
                try:
                    cache_ttl = ttl or settings.embedding_cache_ttl
                    redis_client.setex(key, cache_ttl, json.dumps(result))
                    logger.debug(f"Cached embedding: {key}")
                except Exception as e:
                    logger.warning(f"Cache set failed: {e}")
            
            return result
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def get_embeddings() -> Embeddings:
    """Get embeddings instance based on configuration.
    
    Returns:
        Embeddings instance configured based on available API keys
    """
    if settings.openai_api_key:
        logger.info(f"Using OpenAI embeddings: {settings.openai_embedding_model} with {settings.openai_embedding_dimensions} dimensions")
        return OpenAIEmbeddings(
            api_key=settings.openai_api_key,
            model=settings.openai_embedding_model,
            dimensions=settings.openai_embedding_dimensions,
        )
    elif settings.google_api_key:
        logger.info(f"Using Google embeddings: {settings.google_embedding_model}")
        return GoogleGenerativeAIEmbeddings(
            google_api_key=settings.google_api_key,
            model=settings.google_embedding_model,
        )
    else:
        raise ValueError("No embedding API key configured")