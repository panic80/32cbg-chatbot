"""Dependency injection container for service management.

This module provides a centralized container for managing service instances,
their lifecycle, and dependencies. It replaces global state with a proper
dependency injection pattern.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, TYPE_CHECKING

from app.core.logging import get_logger

if TYPE_CHECKING:
    from app.core.interfaces import (
        ICacheService,
        IVectorStoreManager,
        IDocumentStore,
        ISourceRepository,
        IQueryLogger,
    )
    from app.core.config import Settings

logger = get_logger(__name__)


class ServiceNotInitializedError(Exception):
    """Raised when accessing a service that hasn't been initialized."""

    def __init__(self, service_name: str):
        super().__init__(f"Service '{service_name}' has not been initialized. "
                         f"Call container.initialize() first.")
        self.service_name = service_name


@dataclass
class ServiceContainer:
    """Centralized container for dependency injection.

    This container manages all service instances and their lifecycle.
    Services are lazily accessed through properties that raise
    ServiceNotInitializedError if accessed before initialization.

    Usage:
        container = ServiceContainer()
        await container.initialize(settings)

        # Access services
        cache = container.cache_service
        vector_store = container.vector_store_manager

        # Cleanup
        await container.shutdown()
    """

    _cache_service: Optional[ICacheService] = field(default=None, repr=False)
    _vector_store_manager: Optional[IVectorStoreManager] = field(default=None, repr=False)
    _document_store: Optional[IDocumentStore] = field(default=None, repr=False)
    _source_repository: Optional[ISourceRepository] = field(default=None, repr=False)
    _query_logger: Optional[IQueryLogger] = field(default=None, repr=False)
    _retrieval_pipeline_cache: Dict[str, Any] = field(default_factory=dict, repr=False)
    _initialized: bool = field(default=False, repr=True)
    _settings: Optional[Settings] = field(default=None, repr=False)

    async def initialize(self, settings: Settings) -> None:
        """Initialize all services.

        Args:
            settings: Application settings.

        Raises:
            Exception: If any service fails to initialize.
        """
        if self._initialized:
            logger.warning("Container already initialized, skipping")
            return

        self._settings = settings
        logger.info("Initializing service container...")

        try:
            # Import here to avoid circular imports
            from app.services.cache import CacheService
            from app.core.vectorstore import VectorStoreManager
            from app.services.document_store import DocumentStore
            from app.services.source_repository import SourceRepository
            from app.services.query_logger import query_logger
            from app.core.langchain_config import LangChainConfig

            # Initialize LangChain configuration
            LangChainConfig.initialize()
            logger.info("LangChain configuration initialized")

            # Initialize cache service
            self._cache_service = CacheService()
            await self._cache_service.connect()
            logger.info("Cache service initialized")

            # Initialize vector store
            self._vector_store_manager = VectorStoreManager()
            await self._vector_store_manager.initialize()
            logger.info("Vector store initialized")

            # Initialize source repository
            self._source_repository = SourceRepository()
            await self._source_repository.initialize()
            logger.info("Source repository initialized")

            # Initialize document store (depends on vector store and cache)
            self._document_store = DocumentStore(
                self._vector_store_manager,
                self._cache_service,
                source_repository=self._source_repository
            )
            logger.info("Document store initialized")

            # Initialize query logger
            self._query_logger = query_logger
            await self._query_logger.initialize()
            logger.info("Query logger initialized")

            # Initialize LLM pool if enabled
            if settings.enable_llm_pool:
                from app.services.llm_pool import initialize_llm_pool
                await initialize_llm_pool()
                logger.info("LLM connection pool initialized")

            self._initialized = True
            logger.info("Service container initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize service container: {e}")
            await self.shutdown()
            raise

    async def shutdown(self) -> None:
        """Shutdown all services and cleanup resources."""
        logger.info("Shutting down service container...")

        # Shutdown LLM pool if enabled
        if self._settings and self._settings.enable_llm_pool:
            try:
                from app.services.llm_pool import shutdown_llm_pool
                await shutdown_llm_pool()
                logger.info("LLM connection pool shut down")
            except Exception as e:
                logger.error(f"Error shutting down LLM pool: {e}")

        # Disconnect cache
        if self._cache_service:
            try:
                await self._cache_service.disconnect()
                logger.info("Cache service disconnected")
            except Exception as e:
                logger.error(f"Error disconnecting cache: {e}")

        # Close vector store
        if self._vector_store_manager:
            try:
                await self._vector_store_manager.close()
                logger.info("Vector store closed")
            except Exception as e:
                logger.error(f"Error closing vector store: {e}")

        # Clear pipeline cache
        self._retrieval_pipeline_cache.clear()

        self._initialized = False
        logger.info("Service container shut down")

    @property
    def is_initialized(self) -> bool:
        """Check if the container is initialized."""
        return self._initialized

    @property
    def settings(self) -> Settings:
        """Get application settings."""
        if self._settings is None:
            raise ServiceNotInitializedError("settings")
        return self._settings

    @property
    def cache_service(self) -> ICacheService:
        """Get the cache service instance."""
        if self._cache_service is None:
            raise ServiceNotInitializedError("cache_service")
        return self._cache_service

    @property
    def vector_store_manager(self) -> IVectorStoreManager:
        """Get the vector store manager instance."""
        if self._vector_store_manager is None:
            raise ServiceNotInitializedError("vector_store_manager")
        return self._vector_store_manager

    @property
    def document_store(self) -> IDocumentStore:
        """Get the document store instance."""
        if self._document_store is None:
            raise ServiceNotInitializedError("document_store")
        return self._document_store

    @property
    def source_repository(self) -> ISourceRepository:
        """Get the source repository instance."""
        if self._source_repository is None:
            raise ServiceNotInitializedError("source_repository")
        return self._source_repository

    @property
    def query_logger(self) -> IQueryLogger:
        """Get the query logger instance."""
        if self._query_logger is None:
            raise ServiceNotInitializedError("query_logger")
        return self._query_logger

    @property
    def retrieval_pipeline_cache(self) -> Dict[str, Any]:
        """Get the retrieval pipeline cache."""
        return self._retrieval_pipeline_cache

    def set_cache_service(self, service: ICacheService) -> None:
        """Set the cache service (for testing)."""
        self._cache_service = service

    def set_vector_store_manager(self, manager: IVectorStoreManager) -> None:
        """Set the vector store manager (for testing)."""
        self._vector_store_manager = manager

    def set_document_store(self, store: IDocumentStore) -> None:
        """Set the document store (for testing)."""
        self._document_store = store

    def set_source_repository(self, repo: ISourceRepository) -> None:
        """Set the source repository (for testing)."""
        self._source_repository = repo

    def set_query_logger(self, logger_instance: IQueryLogger) -> None:
        """Set the query logger (for testing)."""
        self._query_logger = logger_instance


# Module-level container instance
_container: Optional[ServiceContainer] = None


def get_container() -> ServiceContainer:
    """Get the global service container instance.

    Returns:
        The global ServiceContainer instance.

    Raises:
        RuntimeError: If the container hasn't been created yet.
    """
    global _container
    if _container is None:
        raise RuntimeError(
            "Service container not created. Call set_container() first "
            "or use the FastAPI app.state.container."
        )
    return _container


def set_container(container: ServiceContainer) -> None:
    """Set the global service container instance.

    Args:
        container: The ServiceContainer instance to use globally.
    """
    global _container
    _container = container


def create_container() -> ServiceContainer:
    """Create a new service container instance.

    This is useful for creating isolated containers in tests.

    Returns:
        A new ServiceContainer instance.
    """
    return ServiceContainer()
