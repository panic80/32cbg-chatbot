"""Tests for the unified cache key generator."""

import pytest

from app.services.unified_cache.key_generator import CacheKeyGenerator


class TestEmbeddingKeys:
    """Tests for embedding key generation."""

    def test_embedding_key_format(self):
        """Test embedding key has correct format."""
        key = CacheKeyGenerator.embedding("test text", "text-embedding-3-small")
        assert key.startswith("emb:")
        assert "v1" in key
        assert "text-embedding-3-small" in key

    def test_embedding_key_deterministic(self):
        """Test same input produces same key."""
        key1 = CacheKeyGenerator.embedding("hello world", "model-a")
        key2 = CacheKeyGenerator.embedding("hello world", "model-a")
        assert key1 == key2

    def test_embedding_key_different_text(self):
        """Test different text produces different keys."""
        key1 = CacheKeyGenerator.embedding("hello", "model-a")
        key2 = CacheKeyGenerator.embedding("world", "model-a")
        assert key1 != key2

    def test_embedding_key_different_model(self):
        """Test different model produces different keys."""
        key1 = CacheKeyGenerator.embedding("hello", "model-a")
        key2 = CacheKeyGenerator.embedding("hello", "model-b")
        assert key1 != key2

    def test_embedding_key_default_model(self):
        """Test default model is used when not specified."""
        key = CacheKeyGenerator.embedding("test")
        assert "default" in key


class TestQueryKeys:
    """Tests for query key generation."""

    def test_query_key_format(self):
        """Test query key has correct format."""
        key = CacheKeyGenerator.query("what is the rate?")
        assert key.startswith("qry:")
        assert "v1" in key

    def test_query_key_with_filters(self):
        """Test query key includes filters."""
        key1 = CacheKeyGenerator.query("test", {"category": "travel"})
        key2 = CacheKeyGenerator.query("test", {"category": "lodging"})
        assert key1 != key2

    def test_query_key_filter_order_independent(self):
        """Test filter order doesn't affect key."""
        key1 = CacheKeyGenerator.query("test", {"a": 1, "b": 2})
        key2 = CacheKeyGenerator.query("test", {"b": 2, "a": 1})
        assert key1 == key2

    def test_query_key_none_filters(self):
        """Test query key with None filters."""
        key1 = CacheKeyGenerator.query("test", None)
        key2 = CacheKeyGenerator.query("test")
        assert key1 == key2


class TestRetrievalKeys:
    """Tests for retrieval key generation."""

    def test_retrieval_key_format(self):
        """Test retrieval key has correct format."""
        key = CacheKeyGenerator.retrieval(
            "test query",
            ["dense", "sparse"],
            60
        )
        assert key.startswith("ret:")
        assert "v1" in key
        assert "k60" in key

    def test_retrieval_key_retriever_order_independent(self):
        """Test retriever order doesn't affect key."""
        key1 = CacheKeyGenerator.retrieval("test", ["sparse", "dense"], 60)
        key2 = CacheKeyGenerator.retrieval("test", ["dense", "sparse"], 60)
        assert key1 == key2

    def test_retrieval_key_different_rrf_k(self):
        """Test different rrf_k produces different keys."""
        key1 = CacheKeyGenerator.retrieval("test", ["dense"], 60)
        key2 = CacheKeyGenerator.retrieval("test", ["dense"], 100)
        assert key1 != key2

    def test_retrieval_key_with_max_docs(self):
        """Test retrieval key includes max_docs."""
        key1 = CacheKeyGenerator.retrieval("test", ["dense"], 60, max_docs=10)
        key2 = CacheKeyGenerator.retrieval("test", ["dense"], 60, max_docs=20)
        assert key1 != key2
        assert "max10" in key1
        assert "max20" in key2

    def test_retrieval_key_index_version(self):
        """Test different index version produces different keys."""
        key1 = CacheKeyGenerator.retrieval("test", ["dense"], 60, index_version="v1")
        key2 = CacheKeyGenerator.retrieval("test", ["dense"], 60, index_version="v2")
        assert key1 != key2


class TestDocumentKeys:
    """Tests for document key generation."""

    def test_document_key_format(self):
        """Test document key has correct format."""
        key = CacheKeyGenerator.document("test query", "hybrid")
        assert key.startswith("doc:")
        assert "v1" in key
        assert "hybrid" in key

    def test_document_key_with_filters(self):
        """Test document key includes filters."""
        key1 = CacheKeyGenerator.document("test", "default", {"source": "A"})
        key2 = CacheKeyGenerator.document("test", "default", {"source": "B"})
        assert key1 != key2


class TestResponseKeys:
    """Tests for response key generation."""

    def test_response_key_format(self):
        """Test response key has correct format."""
        key = CacheKeyGenerator.response("query", "ctx123", "gpt-4")
        assert key.startswith("resp:")
        assert "v1" in key
        assert "gpt-4" in key
        assert "ctx123" in key

    def test_response_key_different_context(self):
        """Test different context produces different keys."""
        key1 = CacheKeyGenerator.response("query", "ctx1", "gpt-4")
        key2 = CacheKeyGenerator.response("query", "ctx2", "gpt-4")
        assert key1 != key2

    def test_response_key_with_params(self):
        """Test response key with additional params."""
        key1 = CacheKeyGenerator.response(
            "query", "ctx", "gpt-4",
            {"temperature": 0.0}
        )
        key2 = CacheKeyGenerator.response(
            "query", "ctx", "gpt-4",
            {"temperature": 0.7}
        )
        assert key1 != key2


class TestClassificationKeys:
    """Tests for classification key generation."""

    def test_classification_key_format(self):
        """Test classification key has correct format."""
        key = CacheKeyGenerator.classification("what is the rate?")
        assert key.startswith("cls:")
        assert "v1" in key

    def test_classification_key_deterministic(self):
        """Test same query produces same key."""
        key1 = CacheKeyGenerator.classification("test query")
        key2 = CacheKeyGenerator.classification("test query")
        assert key1 == key2


class TestGenericKeys:
    """Tests for generic key generation."""

    def test_generic_key_format(self):
        """Test generic key has correct format."""
        key = CacheKeyGenerator.generic("custom", "arg1", "arg2")
        assert key == "custom:v1:arg1:arg2"

    def test_generic_key_with_numbers(self):
        """Test generic key with numeric arguments."""
        key = CacheKeyGenerator.generic("prefix", 1, 2, 3)
        assert key == "prefix:v1:1:2:3"


class TestHashContent:
    """Tests for content hashing."""

    def test_hash_content_deterministic(self):
        """Test same content produces same hash."""
        hash1 = CacheKeyGenerator.hash_content("test content")
        hash2 = CacheKeyGenerator.hash_content("test content")
        assert hash1 == hash2

    def test_hash_content_different(self):
        """Test different content produces different hashes."""
        hash1 = CacheKeyGenerator.hash_content("content A")
        hash2 = CacheKeyGenerator.hash_content("content B")
        assert hash1 != hash2

    def test_hash_content_format(self):
        """Test hash is a valid MD5 hex string."""
        content_hash = CacheKeyGenerator.hash_content("test")
        assert len(content_hash) == 32  # MD5 hex length
        assert all(c in '0123456789abcdef' for c in content_hash)
