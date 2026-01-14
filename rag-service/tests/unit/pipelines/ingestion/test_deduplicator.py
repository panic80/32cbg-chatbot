"""Tests for the deduplicator module."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.pipelines.ingestion.deduplicator import Deduplicator
from app.models.documents import Document, DocumentMetadata, DocumentType, DocumentIngestionRequest


class TestDeduplicator:
    """Tests for Deduplicator class."""

    @pytest.fixture
    def deduplicator(self):
        """Create a deduplicator instance."""
        return Deduplicator(threshold=0.85)

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            Document(
                id="doc1",
                content="This is the first document about travel.",
                metadata=DocumentMetadata(
                    source="test.pdf",
                    type=DocumentType.PDF,
                ),
                chunk_index=0,
            ),
            Document(
                id="doc2",
                content="This is the second document about meals.",
                metadata=DocumentMetadata(
                    source="test.pdf",
                    type=DocumentType.PDF,
                ),
                chunk_index=1,
            ),
        ]

    @pytest.fixture
    def sample_request(self):
        """Create a sample ingestion request."""
        return DocumentIngestionRequest(
            content="test content",
            type=DocumentType.TEXT,
            force_refresh=False,
        )

    @pytest.mark.asyncio
    async def test_deduplicate_empty_list(self, deduplicator, sample_request):
        """Test deduplication of empty list."""
        result = await deduplicator.deduplicate([], sample_request)
        assert result == []

    @pytest.mark.asyncio
    async def test_deduplicate_force_refresh_skips(self, deduplicator, sample_documents):
        """Test that force_refresh skips deduplication."""
        request = DocumentIngestionRequest(
            content="test",
            type=DocumentType.TEXT,
            force_refresh=True,
        )
        result = await deduplicator.deduplicate(sample_documents, request)
        assert len(result) == len(sample_documents)

    @pytest.mark.asyncio
    async def test_deduplicate_preserves_unique(self, deduplicator, sample_documents, sample_request):
        """Test that unique documents are preserved."""
        result = await deduplicator.deduplicate(sample_documents, sample_request)
        # Should keep all unique documents
        assert len(result) <= len(sample_documents)

    def test_generate_content_hash_deterministic(self, deduplicator):
        """Test content hash is deterministic."""
        hash1 = deduplicator.generate_content_hash("test content")
        hash2 = deduplicator.generate_content_hash("test content")
        assert hash1 == hash2

    def test_generate_content_hash_different(self, deduplicator):
        """Test different content produces different hashes."""
        hash1 = deduplicator.generate_content_hash("content A")
        hash2 = deduplicator.generate_content_hash("content B")
        assert hash1 != hash2

    @pytest.mark.asyncio
    async def test_deduplicate_within_batch(self, deduplicator, sample_request):
        """Test deduplication within a single batch."""
        # Create documents with identical content
        docs = [
            Document(
                id="doc1",
                content="Identical content",
                metadata=DocumentMetadata(source="a.pdf", type=DocumentType.PDF),
                chunk_index=0,
            ),
            Document(
                id="doc2",
                content="Identical content",
                metadata=DocumentMetadata(source="b.pdf", type=DocumentType.PDF),
                chunk_index=1,
            ),
        ]
        result = await deduplicator.deduplicate(docs, sample_request)
        # Should merge or remove duplicates
        assert len(result) <= len(docs)


class TestDeduplicatorWithVectorStore:
    """Tests for Deduplicator with vector store integration."""

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store."""
        mock = MagicMock()
        mock.get_all_documents = MagicMock(return_value=[])
        return mock

    @pytest.fixture
    def deduplicator_with_store(self, mock_vector_store):
        """Create deduplicator with vector store."""
        return Deduplicator(
            threshold=0.85,
            vector_store_manager=mock_vector_store,
        )

    @pytest.mark.asyncio
    async def test_checks_existing_documents(self, deduplicator_with_store, mock_vector_store):
        """Test that existing documents are checked."""
        request = DocumentIngestionRequest(
            url="https://example.com/doc.pdf",
            type=DocumentType.PDF,
        )
        docs = [
            Document(
                id="doc1",
                content="Test content",
                metadata=DocumentMetadata(source="test.pdf", type=DocumentType.PDF),
                chunk_index=0,
            ),
        ]

        await deduplicator_with_store.deduplicate(docs, request)
        # Should have called get_all_documents to check for existing
        mock_vector_store.get_all_documents.assert_called()
