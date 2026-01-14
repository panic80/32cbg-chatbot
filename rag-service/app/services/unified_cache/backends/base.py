"""Base interface for cache backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Any, List, Dict, AsyncContextManager
from contextlib import asynccontextmanager


@dataclass
class CacheStats:
    """Statistics for cache performance monitoring."""

    hits: int = 0
    misses: int = 0
    errors: int = 0
    evictions: int = 0

    @property
    def total_requests(self) -> int:
        """Total number of cache requests."""
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests

    def record_hit(self) -> None:
        """Record a cache hit."""
        self.hits += 1

    def record_miss(self) -> None:
        """Record a cache miss."""
        self.misses += 1

    def record_error(self) -> None:
        """Record a cache error."""
        self.errors += 1

    def reset(self) -> None:
        """Reset all statistics."""
        self.hits = 0
        self.misses = 0
        self.errors = 0
        self.evictions = 0


class ICacheBackend(ABC):
    """Abstract base class for cache backends.

    All cache backends must implement this interface to work with
    the LayeredCacheService.
    """

    @property
    @abstractmethod
    def enabled(self) -> bool:
        """Check if the backend is enabled and connected."""
        ...

    @property
    @abstractmethod
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        ...

    @abstractmethod
    async def connect(self) -> None:
        """Connect to the cache backend."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the cache backend."""
        ...

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache.

        Args:
            key: The cache key.

        Returns:
            The cached value, or None if not found.
        """
        ...

    @abstractmethod
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """Set a value in the cache.

        Args:
            key: The cache key.
            value: The value to cache (will be JSON serialized).
            ttl: Time-to-live in seconds (None for no expiration).

        Returns:
            True if successful, False otherwise.
        """
        ...

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete a key from the cache.

        Args:
            key: The cache key to delete.

        Returns:
            True if deleted, False if key didn't exist.
        """
        ...

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if a key exists in the cache.

        Args:
            key: The cache key.

        Returns:
            True if the key exists, False otherwise.
        """
        ...

    @abstractmethod
    async def clear_all(self) -> bool:
        """Clear all keys from the cache.

        Returns:
            True if successful, False otherwise.
        """
        ...

    @abstractmethod
    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values from the cache.

        Args:
            keys: List of cache keys.

        Returns:
            Dictionary mapping keys to their values (missing keys are omitted).
        """
        ...

    @abstractmethod
    async def set_many(
        self,
        items: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> int:
        """Set multiple values in the cache.

        Args:
            items: Dictionary mapping keys to values.
            ttl: Time-to-live in seconds (None for no expiration).

        Returns:
            Number of keys successfully set.
        """
        ...

    @abstractmethod
    @asynccontextmanager
    async def pipeline(self) -> AsyncContextManager:
        """Create a pipeline for batch operations.

        Usage:
            async with backend.pipeline() as pipe:
                pipe.get("key1")
                pipe.get("key2")
                results = await pipe.execute()

        Yields:
            A pipeline object for batching operations.
        """
        ...
