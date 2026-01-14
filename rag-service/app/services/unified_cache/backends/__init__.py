"""Cache backend implementations.

Provides different storage backends for the unified cache system:
- RedisBackend: Production Redis-based caching
- MemoryBackend: In-memory caching for testing
"""

from app.services.unified_cache.backends.base import ICacheBackend, CacheStats
from app.services.unified_cache.backends.redis_backend import RedisBackend
from app.services.unified_cache.backends.memory_backend import MemoryBackend

__all__ = [
    "ICacheBackend",
    "CacheStats",
    "RedisBackend",
    "MemoryBackend",
]
