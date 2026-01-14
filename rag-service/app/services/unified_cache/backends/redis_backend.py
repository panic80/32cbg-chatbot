"""Redis cache backend implementation."""

import json
from typing import Optional, Any, List, Dict
from contextlib import asynccontextmanager
from datetime import timedelta

import redis.asyncio as redis

from app.services.unified_cache.backends.base import ICacheBackend, CacheStats
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class RedisPipeline:
    """Wrapper for Redis pipeline operations."""

    def __init__(self, pipeline, stats: CacheStats):
        self._pipeline = pipeline
        self._stats = stats
        self._operations: List[str] = []

    def get(self, key: str) -> "RedisPipeline":
        """Queue a GET operation."""
        self._pipeline.get(key)
        self._operations.append("get")
        return self

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> "RedisPipeline":
        """Queue a SET operation."""
        serialized = json.dumps(value)
        if ttl:
            self._pipeline.setex(key, timedelta(seconds=ttl), serialized)
        else:
            self._pipeline.set(key, serialized)
        self._operations.append("set")
        return self

    def delete(self, key: str) -> "RedisPipeline":
        """Queue a DELETE operation."""
        self._pipeline.delete(key)
        self._operations.append("delete")
        return self

    async def execute(self) -> List[Any]:
        """Execute all queued operations."""
        results = await self._pipeline.execute()

        # Process results and update stats
        processed = []
        for op, result in zip(self._operations, results):
            if op == "get":
                if result is not None:
                    self._stats.record_hit()
                    try:
                        processed.append(json.loads(result))
                    except (json.JSONDecodeError, TypeError):
                        processed.append(result)
                else:
                    self._stats.record_miss()
                    processed.append(None)
            else:
                processed.append(result)

        return processed


class RedisBackend(ICacheBackend):
    """Redis-based cache backend.

    Provides a production-ready cache backend using Redis with:
    - Automatic JSON serialization/deserialization
    - Pipeline support for batch operations
    - Statistics tracking
    - Graceful error handling
    """

    def __init__(self, redis_url: Optional[str] = None):
        """Initialize Redis backend.

        Args:
            redis_url: Redis connection URL. If not provided, uses settings.redis_url.
        """
        self._redis_url = redis_url or settings.redis_url
        self._client: Optional[redis.Redis] = None
        self._enabled = bool(self._redis_url)
        self._stats = CacheStats()

    @property
    def enabled(self) -> bool:
        """Check if Redis is enabled and connected."""
        return self._enabled and self._client is not None

    @property
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats

    @property
    def client(self) -> Optional[redis.Redis]:
        """Get the Redis client (for advanced operations)."""
        return self._client

    async def connect(self) -> None:
        """Connect to Redis."""
        if not self._redis_url:
            logger.info("Redis URL not configured, cache disabled")
            self._enabled = False
            return

        try:
            self._client = redis.from_url(
                self._redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            # Test connection
            await self._client.ping()
            self._enabled = True
            logger.info("Connected to Redis cache")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self._enabled = False
            self._client = None

    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self._client:
            await self._client.close()
            self._client = None
            logger.info("Disconnected from Redis cache")

    async def get(self, key: str) -> Optional[Any]:
        """Get a value from Redis."""
        if not self.enabled:
            return None

        try:
            value = await self._client.get(key)
            if value is not None:
                self._stats.record_hit()
                return json.loads(value)
            else:
                self._stats.record_miss()
                return None
        except Exception as e:
            logger.error(f"Redis GET error for key {key}: {e}")
            self._stats.record_error()
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """Set a value in Redis."""
        if not self.enabled:
            return False

        try:
            serialized = json.dumps(value)
            if ttl:
                await self._client.setex(
                    key,
                    timedelta(seconds=ttl),
                    serialized
                )
            else:
                await self._client.set(key, serialized)
            return True
        except Exception as e:
            logger.error(f"Redis SET error for key {key}: {e}")
            self._stats.record_error()
            return False

    async def delete(self, key: str) -> bool:
        """Delete a key from Redis."""
        if not self.enabled:
            return False

        try:
            result = await self._client.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"Redis DELETE error for key {key}: {e}")
            self._stats.record_error()
            return False

    async def exists(self, key: str) -> bool:
        """Check if a key exists in Redis."""
        if not self.enabled:
            return False

        try:
            return await self._client.exists(key) > 0
        except Exception as e:
            logger.error(f"Redis EXISTS error for key {key}: {e}")
            self._stats.record_error()
            return False

    async def clear_all(self) -> bool:
        """Clear all keys from Redis (flushdb)."""
        if not self.enabled:
            return True

        try:
            await self._client.flushdb()
            logger.info("Cleared all Redis cache entries")
            return True
        except Exception as e:
            logger.error(f"Redis FLUSHDB error: {e}")
            self._stats.record_error()
            return False

    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values from Redis using MGET."""
        if not self.enabled or not keys:
            return {}

        try:
            values = await self._client.mget(keys)
            results = {}

            for key, value in zip(keys, values):
                if value is not None:
                    self._stats.record_hit()
                    try:
                        results[key] = json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        results[key] = value
                else:
                    self._stats.record_miss()

            return results
        except Exception as e:
            logger.error(f"Redis MGET error: {e}")
            self._stats.record_error()
            return {}

    async def set_many(
        self,
        items: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> int:
        """Set multiple values in Redis using pipeline."""
        if not self.enabled or not items:
            return 0

        try:
            async with self._client.pipeline() as pipe:
                for key, value in items.items():
                    serialized = json.dumps(value)
                    if ttl:
                        pipe.setex(key, timedelta(seconds=ttl), serialized)
                    else:
                        pipe.set(key, serialized)
                await pipe.execute()
            return len(items)
        except Exception as e:
            logger.error(f"Redis pipeline SET error: {e}")
            self._stats.record_error()
            return 0

    @asynccontextmanager
    async def pipeline(self):
        """Create a Redis pipeline for batch operations."""
        if not self.enabled:
            # Return a no-op pipeline
            yield _NoOpPipeline()
            return

        async with self._client.pipeline() as pipe:
            yield RedisPipeline(pipe, self._stats)


class _NoOpPipeline:
    """No-op pipeline when Redis is disabled."""

    def get(self, key: str) -> "_NoOpPipeline":
        return self

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> "_NoOpPipeline":
        return self

    def delete(self, key: str) -> "_NoOpPipeline":
        return self

    async def execute(self) -> List[Any]:
        return []
