"""Tests for the unified layered cache service."""

import pytest
import time
import asyncio

from app.services.unified_cache.layered_cache import LayeredCacheService, CacheConfig
from app.services.unified_cache.backends.memory_backend import MemoryBackend


class TestL1EmbeddingCache:
    """Tests for L1 embedding cache layer."""

    @pytest.mark.asyncio
    async def test_set_and_get_embedding(self):
        """Test setting and getting an embedding."""
        backend = MemoryBackend()
        await backend.connect()
        cache = LayeredCacheService(backend)

        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

        result = await cache.set_embedding("test text", embedding)
        assert result is True

        retrieved = await cache.get_embedding("test text")
        assert retrieved == embedding

    @pytest.mark.asyncio
    async def test_get_missing_embedding(self):
        """Test getting a non-existent embedding returns None."""
        backend = MemoryBackend()
        await backend.connect()
        cache = LayeredCacheService(backend)

        result = await cache.get_embedding("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_embedding_model_isolation(self):
        """Test embeddings are isolated by model."""
        backend = MemoryBackend()
        await backend.connect()
        cache = LayeredCacheService(backend)

        embedding_a = [0.1, 0.2, 0.3]
        embedding_b = [0.4, 0.5, 0.6]

        await cache.set_embedding("text", embedding_a, model="model-a")
        await cache.set_embedding("text", embedding_b, model="model-b")

        retrieved_a = await cache.get_embedding("text", model="model-a")
        retrieved_b = await cache.get_embedding("text", model="model-b")

        assert retrieved_a == embedding_a
        assert retrieved_b == embedding_b

    @pytest.mark.asyncio
    async def test_get_embeddings_batch(self):
        """Test batch embedding retrieval."""
        backend = MemoryBackend()
        await backend.connect()
        cache = LayeredCacheService(backend)

        # Set some embeddings
        await cache.set_embedding("text1", [0.1, 0.2])
        await cache.set_embedding("text2", [0.3, 0.4])

        # Get batch (including one miss)
        results = await cache.get_embeddings_batch(["text1", "text2", "text3"])

        assert len(results) == 2
        assert results["text1"] == [0.1, 0.2]
        assert results["text2"] == [0.3, 0.4]
        assert "text3" not in results

    @pytest.mark.asyncio
    async def test_set_embeddings_batch(self):
        """Test batch embedding setting."""
        backend = MemoryBackend()
        await backend.connect()
        cache = LayeredCacheService(backend)

        embeddings = {
            "text1": [0.1, 0.2],
            "text2": [0.3, 0.4],
            "text3": [0.5, 0.6],
        }

        count = await cache.set_embeddings_batch(embeddings)
        assert count == 3

        # Verify all were set
        for text, embedding in embeddings.items():
            retrieved = await cache.get_embedding(text)
            assert retrieved == embedding


class TestL2RetrievalCache:
    """Tests for L2 retrieval cache layer."""

    @pytest.mark.asyncio
    async def test_set_and_get_retrieval_results(self):
        """Test setting and getting retrieval results."""
        backend = MemoryBackend()
        await backend.connect()
        cache = LayeredCacheService(backend)

        results = [
            {"content": "doc1", "score": 0.9},
            {"content": "doc2", "score": 0.8},
        ]

        success = await cache.set_retrieval_results(
            query="test query",
            retriever_names=["dense", "sparse"],
            rrf_k=60,
            results=results
        )
        assert success is True

        retrieved = await cache.get_retrieval_results(
            query="test query",
            retriever_names=["dense", "sparse"],
            rrf_k=60
        )
        assert retrieved == results

    @pytest.mark.asyncio
    async def test_retrieval_cache_miss(self):
        """Test retrieval cache miss returns None."""
        backend = MemoryBackend()
        await backend.connect()
        cache = LayeredCacheService(backend)

        result = await cache.get_retrieval_results(
            query="nonexistent",
            retriever_names=["dense"],
            rrf_k=60
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_retrieval_different_params(self):
        """Test different params produce cache misses."""
        backend = MemoryBackend()
        await backend.connect()
        cache = LayeredCacheService(backend)

        results = [{"content": "doc1"}]

        await cache.set_retrieval_results(
            query="test",
            retriever_names=["dense"],
            rrf_k=60,
            results=results
        )

        # Different rrf_k should miss
        retrieved = await cache.get_retrieval_results(
            query="test",
            retriever_names=["dense"],
            rrf_k=100
        )
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_set_and_get_documents(self):
        """Test setting and getting documents."""
        backend = MemoryBackend()
        await backend.connect()
        cache = LayeredCacheService(backend)

        docs = [{"id": "1", "content": "test"}]

        await cache.set_documents("query", docs, "hybrid")

        retrieved = await cache.get_documents("query", "hybrid")
        assert retrieved == docs


class TestL3ResponseCache:
    """Tests for L3 response cache layer."""

    @pytest.mark.asyncio
    async def test_set_and_get_response(self):
        """Test setting and getting a response."""
        backend = MemoryBackend()
        await backend.connect()
        cache = LayeredCacheService(backend)

        response = {
            "answer": "The rate is $50",
            "sources": ["doc1", "doc2"]
        }

        success = await cache.set_response(
            query="what is the rate?",
            context_hash="ctx123",
            model="gpt-4",
            response=response
        )
        assert success is True

        retrieved = await cache.get_response(
            query="what is the rate?",
            context_hash="ctx123",
            model="gpt-4"
        )
        assert retrieved == response

    @pytest.mark.asyncio
    async def test_response_different_context(self):
        """Test different context produces different cache entries."""
        backend = MemoryBackend()
        await backend.connect()
        cache = LayeredCacheService(backend)

        response1 = {"answer": "answer 1"}
        response2 = {"answer": "answer 2"}

        await cache.set_response("query", "ctx1", "gpt-4", response1)
        await cache.set_response("query", "ctx2", "gpt-4", response2)

        retrieved1 = await cache.get_response("query", "ctx1", "gpt-4")
        retrieved2 = await cache.get_response("query", "ctx2", "gpt-4")

        assert retrieved1 == response1
        assert retrieved2 == response2


class TestClassificationCache:
    """Tests for query classification cache."""

    @pytest.mark.asyncio
    async def test_set_and_get_classification(self):
        """Test setting and getting a classification."""
        backend = MemoryBackend()
        await backend.connect()
        cache = LayeredCacheService(backend)

        classification = {
            "type": "rate_inquiry",
            "confidence": 0.95
        }

        await cache.set_classification("what is the rate?", classification)

        retrieved = await cache.get_classification("what is the rate?")
        assert retrieved == classification


class TestCacheStats:
    """Tests for cache statistics."""

    @pytest.mark.asyncio
    async def test_stats_track_hits_and_misses(self):
        """Test that stats track hits and misses."""
        backend = MemoryBackend()
        await backend.connect()
        cache = LayeredCacheService(backend)

        # Initial stats
        stats = cache.stats
        assert stats.l1_embeddings.hits == 0
        assert stats.l1_embeddings.misses == 0

        # Miss
        await cache.get_embedding("nonexistent")
        assert cache.stats.l1_embeddings.misses == 1

        # Set then hit
        await cache.set_embedding("test", [0.1, 0.2])
        await cache.get_embedding("test")
        assert cache.stats.l1_embeddings.hits == 1

    @pytest.mark.asyncio
    async def test_stats_by_layer(self):
        """Test stats are tracked per layer."""
        backend = MemoryBackend()
        await backend.connect()
        cache = LayeredCacheService(backend)

        # L1 operations
        await cache.get_embedding("test")  # miss
        await cache.set_embedding("test", [0.1])
        await cache.get_embedding("test")  # hit

        # L2 operations
        await cache.get_documents("query")  # miss

        # L3 operations
        await cache.get_response("q", "ctx", "model")  # miss

        stats = cache.stats

        assert stats.l1_embeddings.hits == 1
        assert stats.l1_embeddings.misses == 1

        assert stats.l2_retrieval.hits == 0
        assert stats.l2_retrieval.misses == 1

        assert stats.l3_responses.hits == 0
        assert stats.l3_responses.misses == 1

    @pytest.mark.asyncio
    async def test_overall_hit_rate(self):
        """Test overall hit rate calculation."""
        backend = MemoryBackend()
        await backend.connect()
        cache = LayeredCacheService(backend)

        # Set up some data
        await cache.set_embedding("a", [0.1])
        await cache.set_embedding("b", [0.2])

        # 2 hits, 1 miss in L1
        await cache.get_embedding("a")  # hit
        await cache.get_embedding("b")  # hit
        await cache.get_embedding("c")  # miss

        # 1 miss in L2
        await cache.get_documents("query")  # miss

        stats = cache.stats
        assert stats.total_hits == 2
        assert stats.total_misses == 2  # 1 in L1, 1 in L2
        assert stats.overall_hit_rate == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_reset_stats(self):
        """Test resetting statistics."""
        backend = MemoryBackend()
        await backend.connect()
        cache = LayeredCacheService(backend)

        await cache.get_embedding("test")
        assert cache.stats.l1_embeddings.misses == 1

        cache.reset_stats()

        assert cache.stats.l1_embeddings.misses == 0
        assert cache.stats.l1_embeddings.hits == 0


class TestGenericOperations:
    """Tests for generic cache operations."""

    @pytest.mark.asyncio
    async def test_generic_get_set(self):
        """Test generic get and set operations."""
        backend = MemoryBackend()
        await backend.connect()
        cache = LayeredCacheService(backend)

        await cache.set("custom:key", {"data": "value"})

        result = await cache.get("custom:key")
        assert result == {"data": "value"}

    @pytest.mark.asyncio
    async def test_delete(self):
        """Test deleting a key."""
        backend = MemoryBackend()
        await backend.connect()
        cache = LayeredCacheService(backend)

        await cache.set_embedding("test", [0.1])

        # Verify it exists
        result = await cache.get_embedding("test")
        assert result is not None

        # Delete using the full key
        from app.services.unified_cache.key_generator import CacheKeyGenerator
        key = CacheKeyGenerator.embedding("test", "default")
        deleted = await cache.delete(key)
        assert deleted is True

        # Verify it's gone
        result = await cache.get_embedding("test")
        assert result is None

    @pytest.mark.asyncio
    async def test_clear_all(self):
        """Test clearing all cache entries."""
        backend = MemoryBackend()
        await backend.connect()
        cache = LayeredCacheService(backend)

        await cache.set_embedding("test1", [0.1])
        await cache.set_embedding("test2", [0.2])
        await cache.set_documents("query", [{"id": "1"}])

        result = await cache.clear_all()
        assert result is True

        # Verify all cleared
        assert await cache.get_embedding("test1") is None
        assert await cache.get_embedding("test2") is None
        assert await cache.get_documents("query") is None


class TestCacheConfig:
    """Tests for cache configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CacheConfig()

        assert config.embedding_ttl == 604800  # 7 days
        assert config.retrieval_ttl == 86400  # 24 hours
        assert config.response_ttl == 21600  # 6 hours
        assert config.classification_ttl == 3600  # 1 hour

    def test_custom_config(self):
        """Test custom configuration values."""
        config = CacheConfig(
            embedding_ttl=1000,
            retrieval_ttl=2000,
            response_ttl=3000
        )

        assert config.embedding_ttl == 1000
        assert config.retrieval_ttl == 2000
        assert config.response_ttl == 3000

    @pytest.mark.asyncio
    async def test_config_affects_ttl(self):
        """Test that config TTL is used."""
        backend = MemoryBackend()
        await backend.connect()
        config = CacheConfig(embedding_ttl=1)  # 1 second
        cache = LayeredCacheService(backend, config)

        await cache.set_embedding("test", [0.1])

        # Should exist immediately
        result = await cache.get_embedding("test")
        assert result == [0.1]

        # Wait for TTL to expire
        time.sleep(1.5)

        # Should be expired
        result = await cache.get_embedding("test")
        assert result is None


class TestDisabledCache:
    """Tests for cache when backend is disabled."""

    @pytest.mark.asyncio
    async def test_operations_fail_gracefully(self):
        """Test operations return None/False when cache is disabled."""
        backend = MemoryBackend()
        # Don't connect - cache is disabled
        cache = LayeredCacheService(backend)

        assert cache.enabled is False

        # Operations should fail gracefully
        result = await cache.set_embedding("test", [0.1])
        assert result is False

        result = await cache.get_embedding("test")
        assert result is None
