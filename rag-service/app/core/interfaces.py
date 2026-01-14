"""Protocol definitions for service interfaces.

This module defines the interfaces (Protocols) for all core services
to enable dependency injection and testability.
"""

from typing import Protocol, Optional, List, Dict, Any, Tuple, runtime_checkable
from langchain_core.documents import Document as LangchainDocument


@runtime_checkable
class ICacheService(Protocol):
    """Interface for cache operations."""

    enabled: bool

    async def connect(self) -> None:
        """Connect to the cache backend."""
        ...

    async def disconnect(self) -> None:
        """Disconnect from the cache backend."""
        ...

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        ...

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL."""
        ...

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        ...

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        ...

    async def clear_all(self) -> bool:
        """Clear all cache entries."""
        ...

    def make_key(self, prefix: str, *args) -> str:
        """Create cache key from prefix and arguments."""
        ...

    def make_embedding_key(self, text: str) -> str:
        """Create cache key for embeddings."""
        ...

    def make_query_key(self, query: str, filters: Optional[Dict] = None) -> str:
        """Create cache key for query results."""
        ...


@runtime_checkable
class IVectorStoreManager(Protocol):
    """Interface for vector store operations."""

    async def initialize(self) -> None:
        """Initialize embeddings and vector store."""
        ...

    async def close(self) -> None:
        """Close the vector store and cleanup resources."""
        ...

    def get_all_documents(self, refresh: bool = False) -> List[LangchainDocument]:
        """Return all documents from the vector store (cached).

        Args:
            refresh: If True, forces reloading from the collection.

        Returns:
            List of LangChain Document objects representing the entire corpus.
        """
        ...

    async def search(
        self,
        query: str,
        k: int = 4,
        filter_dict: Optional[Dict[str, Any]] = None,
        search_type: str = "similarity"
    ) -> List[Tuple[LangchainDocument, float]]:
        """Search for similar documents.

        Args:
            query: The search query.
            k: Number of results to return.
            filter_dict: Optional metadata filters.
            search_type: Type of search (similarity, mmr).

        Returns:
            List of (document, score) tuples.
        """
        ...

    async def add_documents(
        self,
        documents: List[LangchainDocument],
        batch_size: int = 100
    ) -> List[str]:
        """Add documents to the vector store.

        Args:
            documents: Documents to add.
            batch_size: Batch size for processing.

        Returns:
            List of document IDs.
        """
        ...

    async def delete_documents(self, ids: List[str]) -> bool:
        """Delete documents by ID.

        Args:
            ids: Document IDs to delete.

        Returns:
            True if successful.
        """
        ...


@runtime_checkable
class IDocumentStore(Protocol):
    """Interface for document store operations."""

    async def search(self, request: Any) -> List[Any]:
        """Search for documents.

        Args:
            request: DocumentSearchRequest with query and filters.

        Returns:
            List of DocumentSearchResult objects.
        """
        ...

    async def get_by_id(self, document_id: str) -> Optional[Any]:
        """Get document by ID.

        Args:
            document_id: The document ID.

        Returns:
            Document or None if not found.
        """
        ...

    async def list_documents(
        self,
        skip: int = 0,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None
    ) -> Any:
        """List documents with pagination.

        Args:
            skip: Number of documents to skip.
            limit: Maximum documents to return.
            filters: Optional metadata filters.

        Returns:
            DocumentListResponse with documents and pagination info.
        """
        ...


@runtime_checkable
class ISourceRepository(Protocol):
    """Interface for source repository operations."""

    async def initialize(self) -> None:
        """Ensure the underlying database and tables exist."""
        ...

    async def upsert_entries(self, entries: Any) -> None:
        """Upsert a batch of source entries and related chunk mappings.

        Args:
            entries: Iterable of SourceCatalogEntry objects.
        """
        ...

    async def get_entry(self, source_id: str) -> Optional[Any]:
        """Retrieve a single source entry by ID.

        Args:
            source_id: The source ID.

        Returns:
            SourceCatalogEntry or None if not found.
        """
        ...

    async def list_entries(
        self,
        skip: int = 0,
        limit: int = 100
    ) -> List[Any]:
        """List source entries with pagination.

        Args:
            skip: Number of entries to skip.
            limit: Maximum entries to return.

        Returns:
            List of SourceCatalogEntry objects.
        """
        ...

    async def delete_entry(self, source_id: str) -> bool:
        """Delete a source entry.

        Args:
            source_id: The source ID to delete.

        Returns:
            True if deleted, False if not found.
        """
        ...

    async def record_query_sources(
        self,
        query_id: str,
        sources: List[Tuple[str, int, Optional[str]]]
    ) -> None:
        """Record which sources were used for a query.

        Args:
            query_id: The query ID.
            sources: List of (source_id, rank, snippet) tuples.
        """
        ...


@runtime_checkable
class IQueryLogger(Protocol):
    """Interface for query logging operations."""

    async def initialize(self) -> None:
        """Initialize the query logger."""
        ...

    async def log_query(
        self,
        query: str,
        response: str,
        sources: List[Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log a query and its response.

        Args:
            query: The user query.
            response: The generated response.
            sources: List of sources used.
            metadata: Optional additional metadata.

        Returns:
            The query ID.
        """
        ...

    async def get_query(self, query_id: str) -> Optional[Any]:
        """Get a logged query by ID.

        Args:
            query_id: The query ID.

        Returns:
            Query log entry or None if not found.
        """
        ...
