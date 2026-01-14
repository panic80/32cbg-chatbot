"""Shared test fixtures for RAG service tests."""

import pytest
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

from app.services.unified_cache.backends.memory_backend import MemoryBackend
from app.services.unified_cache.layered_cache import LayeredCacheService, CacheConfig


# ============================================================================
# Cache Fixtures
# ============================================================================

@pytest.fixture
def memory_backend():
    """Create a fresh memory backend for testing."""
    return MemoryBackend()


@pytest.fixture
async def connected_memory_backend(memory_backend):
    """Create a connected memory backend."""
    await memory_backend.connect()
    yield memory_backend
    await memory_backend.disconnect()


@pytest.fixture
async def layered_cache(connected_memory_backend):
    """Create a layered cache service with memory backend."""
    return LayeredCacheService(connected_memory_backend)


@pytest.fixture
def cache_config():
    """Create a test cache configuration."""
    return CacheConfig(
        embedding_ttl=60,
        retrieval_ttl=30,
        response_ttl=15,
        classification_ttl=10,
    )


# ============================================================================
# Mock Service Fixtures
# ============================================================================

@pytest.fixture
def mock_cache_service():
    """Create a mock cache service."""
    mock = AsyncMock()
    mock.enabled = True
    mock.get = AsyncMock(return_value=None)
    mock.set = AsyncMock(return_value=True)
    mock.delete = AsyncMock(return_value=True)
    mock.get_classification = AsyncMock(return_value=None)
    mock.set_classification = AsyncMock(return_value=True)
    return mock


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store manager."""
    mock = MagicMock()
    mock.search = AsyncMock(return_value=[])
    mock.add_documents = AsyncMock()
    mock.get_all_documents = MagicMock(return_value=[])
    mock.vector_store = MagicMock()
    mock.embeddings = MagicMock()
    return mock


@pytest.fixture
def mock_document_store():
    """Create a mock document store."""
    mock = AsyncMock()
    mock.search = AsyncMock(return_value=[])
    mock.get_by_id = AsyncMock(return_value=None)
    mock.get_stats = AsyncMock(return_value={"total_documents": 0})
    return mock


@pytest.fixture
def mock_llm():
    """Create a mock LLM."""
    mock = MagicMock()
    mock.astream = AsyncMock()
    mock.ainvoke = AsyncMock()
    return mock


@pytest.fixture
def mock_query_logger():
    """Create a mock query logger."""
    mock = AsyncMock()
    mock.log_query = AsyncMock()
    mock.get_query_history = AsyncMock(return_value=[])
    return mock


@pytest.fixture
def mock_source_repository():
    """Create a mock source repository."""
    mock = AsyncMock()
    mock.get = AsyncMock(return_value=None)
    mock.list = AsyncMock(return_value=[])
    mock.upsert_entries = AsyncMock()
    mock.record_query_sources = AsyncMock()
    return mock


# ============================================================================
# Container Fixtures
# ============================================================================

@pytest.fixture
def mock_container(
    mock_cache_service,
    mock_vector_store,
    mock_document_store,
    mock_query_logger,
    mock_source_repository,
):
    """Create a mock service container with all services."""
    from app.core.container import ServiceContainer

    container = ServiceContainer()
    container._cache_service = mock_cache_service
    container._vector_store_manager = mock_vector_store
    container._document_store = mock_document_store
    container._query_logger = mock_query_logger
    container._source_repository = mock_source_repository
    container._initialized = True

    return container


# ============================================================================
# Sample Data Fixtures
# ============================================================================

@pytest.fixture
def sample_embedding():
    """Sample embedding vector."""
    return [0.1, 0.2, 0.3, 0.4, 0.5] * 64  # 320-dim vector


@pytest.fixture
def sample_document():
    """Sample document for testing."""
    from langchain_core.documents import Document

    return Document(
        page_content="This is a test document about travel allowances.",
        metadata={
            "source": "test_source.pdf",
            "title": "Test Document",
            "page": 1,
        },
    )


@pytest.fixture
def sample_documents(sample_document):
    """List of sample documents."""
    from langchain_core.documents import Document

    return [
        sample_document,
        Document(
            page_content="Meal allowances are provided for official travel.",
            metadata={"source": "policy.pdf", "title": "Policy Document", "page": 2},
        ),
        Document(
            page_content="Kilometric rates vary by province.",
            metadata={"source": "rates.pdf", "title": "Rate Tables", "page": 1},
        ),
    ]


@pytest.fixture
def sample_chat_request():
    """Sample chat request."""
    from app.models.query import ChatRequest, Provider

    return ChatRequest(
        message="What is the meal allowance for travel?",
        provider=Provider.OPENAI,
        model="gpt-4",
        use_rag=True,
        include_sources=True,
    )


@pytest.fixture
def sample_ingestion_request():
    """Sample document ingestion request."""
    from app.models.documents import DocumentIngestionRequest, DocumentType

    return DocumentIngestionRequest(
        content="This is test content for ingestion.",
        type=DocumentType.TEXT,
        metadata={"source": "test", "title": "Test Document"},
    )


# ============================================================================
# App State Fixtures
# ============================================================================

@pytest.fixture
def mock_app_state(mock_container):
    """Create a mock FastAPI app state."""
    state = MagicMock()
    state.container = mock_container
    state.cache_service = mock_container.cache_service
    state.vector_store_manager = mock_container.vector_store_manager
    state.document_store = mock_container.document_store
    state.query_logger = mock_container.query_logger
    state.source_repository = mock_container.source_repository
    state.retrieval_pipeline_cache = {}
    return state
