"""In-memory cache backend for testing."""

import time
from typing import Optional, Any, List, Dict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from app.services.unified_cache.backends.base import ICacheBackend, CacheStats


@dataclass
class CacheEntry:
    """A cache entry with optional expiration."""

    value: Any
    expires_at: Optional[float] = None

    def is_expired(self) -> bool:
        """Check if the entry has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at


class MemoryPipeline:
    """In-memory pipeline for batch operations."""

    def __init__(self, backend: "MemoryBackend"):
        self._backend = backend
        self._operations: List[tuple] = []

    def get(self, key: str) -> "MemoryPipeline":
        """Queue a GET operation."""
        self._operations.append(("get", key, None, None))
        return self

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> "MemoryPipeline":
        """Queue a SET operation."""
        self._operations.append(("set", key, value, ttl))
        return self

    def delete(self, key: str) -> "MemoryPipeline":
        """Queue a DELETE operation."""
        self._operations.append(("delete", key, None, None))
        return self

    async def execute(self) -> List[Any]:
        """Execute all queued operations."""
        results = []
        for op, key, value, ttl in self._operations:
            if op == "get":
                result = await self._backend.get(key)
                results.append(result)
            elif op == "set":
                result = await self._backend.set(key, value, ttl)
                results.append(result)
            elif op == "delete":
                result = await self._backend.delete(key)
                results.append(result)
        return results


class MemoryBackend(ICacheBackend):
    """In-memory cache backend for testing.

    Provides a fast, in-memory cache that mimics Redis behavior.
    Useful for unit tests and local development without Redis.

    Features:
    - TTL support with automatic expiration
    - Statistics tracking
    - Pipeline support
    - Thread-safe (async operations)
    """

    def __init__(self, auto_cleanup: bool = True):
        """Initialize memory backend.

        Args:
            auto_cleanup: If True, expired entries are cleaned up on access.
        """
        self._storage: Dict[str, CacheEntry] = {}
        self._enabled = False
        self._stats = CacheStats()
        self._auto_cleanup = auto_cleanup

    @property
    def enabled(self) -> bool:
        """Check if the backend is enabled."""
        return self._enabled

    @property
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats

    async def connect(self) -> None:
        """Enable the cache backend."""
        self._enabled = True

    async def disconnect(self) -> None:
        """Disable the cache backend and clear storage."""
        self._enabled = False
        self._storage.clear()

    def _cleanup_expired(self) -> None:
        """Remove expired entries from storage."""
        if not self._auto_cleanup:
            return

        current_time = time.time()
        expired_keys = [
            key for key, entry in self._storage.items()
            if entry.expires_at and current_time > entry.expires_at
        ]

        for key in expired_keys:
            del self._storage[key]
            self._stats.evictions += 1

    async def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache."""
        if not self._enabled:
            return None

        self._cleanup_expired()

        entry = self._storage.get(key)
        if entry is None:
            self._stats.record_miss()
            return None

        if entry.is_expired():
            del self._storage[key]
            self._stats.evictions += 1
            self._stats.record_miss()
            return None

        self._stats.record_hit()
        return entry.value

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """Set a value in the cache."""
        if not self._enabled:
            return False

        expires_at = None
        if ttl is not None:
            expires_at = time.time() + ttl

        self._storage[key] = CacheEntry(value=value, expires_at=expires_at)
        return True

    async def delete(self, key: str) -> bool:
        """Delete a key from the cache."""
        if not self._enabled:
            return False

        if key in self._storage:
            del self._storage[key]
            return True
        return False

    async def exists(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        if not self._enabled:
            return False

        entry = self._storage.get(key)
        if entry is None:
            return False

        if entry.is_expired():
            del self._storage[key]
            return False

        return True

    async def clear_all(self) -> bool:
        """Clear all keys from the cache."""
        self._storage.clear()
        return True

    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values from the cache."""
        if not self._enabled or not keys:
            return {}

        self._cleanup_expired()

        results = {}
        for key in keys:
            entry = self._storage.get(key)
            if entry and not entry.is_expired():
                results[key] = entry.value
                self._stats.record_hit()
            else:
                self._stats.record_miss()

        return results

    async def set_many(
        self,
        items: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> int:
        """Set multiple values in the cache."""
        if not self._enabled or not items:
            return 0

        expires_at = None
        if ttl is not None:
            expires_at = time.time() + ttl

        for key, value in items.items():
            self._storage[key] = CacheEntry(value=value, expires_at=expires_at)

        return len(items)

    @asynccontextmanager
    async def pipeline(self):
        """Create a pipeline for batch operations."""
        yield MemoryPipeline(self)

    # Testing utilities

    def get_all_keys(self) -> List[str]:
        """Get all keys in the cache (testing utility)."""
        self._cleanup_expired()
        return list(self._storage.keys())

    def get_entry_count(self) -> int:
        """Get the number of entries in the cache (testing utility)."""
        self._cleanup_expired()
        return len(self._storage)

    def get_raw_entry(self, key: str) -> Optional[CacheEntry]:
        """Get a raw cache entry (testing utility)."""
        return self._storage.get(key)
