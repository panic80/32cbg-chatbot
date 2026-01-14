"""Tests for the ServiceContainer dependency injection system."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass
from typing import Optional, Any, Dict, List

from app.core.container import (
    ServiceContainer,
    ServiceNotInitializedError,
    get_container,
    set_container,
    create_container,
)
from app.core.interfaces import (
    ICacheService,
    IVectorStoreManager,
    IDocumentStore,
    ISourceRepository,
    IQueryLogger,
)


# Mock implementations for testing
class MockCacheService:
    """Mock cache service for testing."""

    enabled = True

    async def connect(self) -> None:
        pass

    async def disconnect(self) -> None:
        pass

    async def get(self, key: str) -> Optional[Any]:
        return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        return True

    async def delete(self, key: str) -> bool:
        return True

    async def exists(self, key: str) -> bool:
        return False

    async def clear_all(self) -> bool:
        return True

    def make_key(self, prefix: str, *args) -> str:
        return f"{prefix}:{':'.join(str(a) for a in args)}"

    def make_embedding_key(self, text: str) -> str:
        return f"embedding:{hash(text)}"

    def make_query_key(self, query: str, filters: Optional[Dict] = None) -> str:
        return f"query:{hash(query)}"


class MockVectorStoreManager:
    """Mock vector store manager for testing."""

    async def initialize(self) -> None:
        pass

    async def close(self) -> None:
        pass

    def get_all_documents(self, refresh: bool = False) -> List:
        return []

    async def search(self, query: str, k: int = 4, **kwargs) -> List:
        return []

    async def add_documents(self, documents: List, batch_size: int = 100) -> List[str]:
        return []

    async def delete_documents(self, ids: List[str]) -> bool:
        return True


class MockDocumentStore:
    """Mock document store for testing."""

    async def search(self, request: Any) -> List:
        return []

    async def get_by_id(self, document_id: str) -> Optional[Any]:
        return None

    async def list_documents(self, skip: int = 0, limit: int = 100, filters: Optional[Dict] = None) -> Any:
        return {"documents": [], "total": 0}


class MockSourceRepository:
    """Mock source repository for testing."""

    async def initialize(self) -> None:
        pass

    async def upsert_entries(self, entries: Any) -> None:
        pass

    async def get_entry(self, source_id: str) -> Optional[Any]:
        return None

    async def list_entries(self, skip: int = 0, limit: int = 100) -> List:
        return []

    async def delete_entry(self, source_id: str) -> bool:
        return True

    async def record_query_sources(self, query_id: str, sources: List) -> None:
        pass


class MockQueryLogger:
    """Mock query logger for testing."""

    async def initialize(self) -> None:
        pass

    async def log_query(self, query: str, response: str, sources: List, metadata: Optional[Dict] = None) -> str:
        return "test-query-id"

    async def get_query(self, query_id: str) -> Optional[Any]:
        return None


class TestServiceContainer:
    """Tests for ServiceContainer class."""

    def test_create_container(self):
        """Test container creation."""
        container = ServiceContainer()
        assert container is not None
        assert container.is_initialized is False

    def test_service_not_initialized_error(self):
        """Test that accessing uninitialized services raises error."""
        container = ServiceContainer()

        with pytest.raises(ServiceNotInitializedError) as exc_info:
            _ = container.cache_service
        assert "cache_service" in str(exc_info.value)

        with pytest.raises(ServiceNotInitializedError) as exc_info:
            _ = container.vector_store_manager
        assert "vector_store_manager" in str(exc_info.value)

        with pytest.raises(ServiceNotInitializedError) as exc_info:
            _ = container.document_store
        assert "document_store" in str(exc_info.value)

        with pytest.raises(ServiceNotInitializedError) as exc_info:
            _ = container.source_repository
        assert "source_repository" in str(exc_info.value)

        with pytest.raises(ServiceNotInitializedError) as exc_info:
            _ = container.query_logger
        assert "query_logger" in str(exc_info.value)

    def test_set_services_for_testing(self):
        """Test setting services manually for testing."""
        container = ServiceContainer()

        mock_cache = MockCacheService()
        mock_vector_store = MockVectorStoreManager()
        mock_doc_store = MockDocumentStore()
        mock_source_repo = MockSourceRepository()
        mock_query_logger = MockQueryLogger()

        container.set_cache_service(mock_cache)
        container.set_vector_store_manager(mock_vector_store)
        container.set_document_store(mock_doc_store)
        container.set_source_repository(mock_source_repo)
        container.set_query_logger(mock_query_logger)

        assert container.cache_service is mock_cache
        assert container.vector_store_manager is mock_vector_store
        assert container.document_store is mock_doc_store
        assert container.source_repository is mock_source_repo
        assert container.query_logger is mock_query_logger

    def test_retrieval_pipeline_cache(self):
        """Test retrieval pipeline cache access."""
        container = ServiceContainer()
        cache = container.retrieval_pipeline_cache

        assert isinstance(cache, dict)
        assert len(cache) == 0

        # Test adding to cache
        cache["test_key"] = "test_value"
        assert container.retrieval_pipeline_cache["test_key"] == "test_value"


class TestContainerLifecycle:
    """Tests for container lifecycle management."""

    @pytest.mark.asyncio
    async def test_shutdown_cleans_up(self):
        """Test that shutdown cleans up resources."""
        container = ServiceContainer()

        mock_cache = MockCacheService()
        mock_cache.disconnect = AsyncMock()

        mock_vector_store = MockVectorStoreManager()
        mock_vector_store.close = AsyncMock()

        container.set_cache_service(mock_cache)
        container.set_vector_store_manager(mock_vector_store)

        await container.shutdown()

        mock_cache.disconnect.assert_called_once()
        mock_vector_store.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_handles_errors_gracefully(self):
        """Test that shutdown handles individual service errors."""
        container = ServiceContainer()

        mock_cache = MockCacheService()
        mock_cache.disconnect = AsyncMock(side_effect=Exception("Cache error"))

        mock_vector_store = MockVectorStoreManager()
        mock_vector_store.close = AsyncMock()

        container.set_cache_service(mock_cache)
        container.set_vector_store_manager(mock_vector_store)

        # Should not raise, should log error and continue
        await container.shutdown()

        # Vector store should still be closed even if cache fails
        mock_vector_store.close.assert_called_once()


class TestModuleLevelFunctions:
    """Tests for module-level container functions."""

    def test_get_container_raises_when_not_set(self):
        """Test that get_container raises when no container is set."""
        # Clear any existing container
        set_container(None)

        with pytest.raises(RuntimeError) as exc_info:
            get_container()
        assert "not created" in str(exc_info.value)

    def test_set_and_get_container(self):
        """Test setting and getting the global container."""
        container = ServiceContainer()
        set_container(container)

        retrieved = get_container()
        assert retrieved is container

        # Cleanup
        set_container(None)

    def test_create_container_returns_new_instance(self):
        """Test that create_container returns a new instance."""
        container1 = create_container()
        container2 = create_container()

        assert container1 is not container2
        assert isinstance(container1, ServiceContainer)
        assert isinstance(container2, ServiceContainer)


class TestInterfaceCompliance:
    """Tests verifying mock implementations comply with interfaces."""

    def test_mock_cache_service_implements_protocol(self):
        """Test MockCacheService implements ICacheService protocol."""
        mock = MockCacheService()
        assert isinstance(mock, ICacheService)

    def test_mock_vector_store_implements_protocol(self):
        """Test MockVectorStoreManager implements IVectorStoreManager protocol."""
        mock = MockVectorStoreManager()
        assert isinstance(mock, IVectorStoreManager)

    def test_mock_document_store_implements_protocol(self):
        """Test MockDocumentStore implements IDocumentStore protocol."""
        mock = MockDocumentStore()
        assert isinstance(mock, IDocumentStore)

    def test_mock_source_repository_implements_protocol(self):
        """Test MockSourceRepository implements ISourceRepository protocol."""
        mock = MockSourceRepository()
        assert isinstance(mock, ISourceRepository)

    def test_mock_query_logger_implements_protocol(self):
        """Test MockQueryLogger implements IQueryLogger protocol."""
        mock = MockQueryLogger()
        assert isinstance(mock, IQueryLogger)
