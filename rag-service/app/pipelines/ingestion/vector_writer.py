"""Vector store writing for ingestion pipeline."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document as LangchainDocument

from app.core.config import settings
from app.core.errors import StorageError
from app.core.logging import get_logger
from app.models.documents import Document
from app.pipelines.parallel_ingestion import OptimizedVectorStoreWriter
from app.utils.retry import RetryManager, AGGRESSIVE_RETRY_CONFIG

logger = get_logger(__name__)


class VectorWriter:
    """Writes documents to vector store with parallel embedding generation."""

    def __init__(
        self,
        vector_store_manager: Any,
        embedding_cache: Optional[Any] = None,
        retry_manager: Optional[RetryManager] = None,
    ):
        """Initialize vector writer.

        Args:
            vector_store_manager: The vector store manager.
            embedding_cache: Optional embedding cache service.
            retry_manager: Optional retry manager for failures.
        """
        self._vector_store = vector_store_manager
        self._embedding_cache = embedding_cache
        self._retry_manager = retry_manager or RetryManager(AGGRESSIVE_RETRY_CONFIG)
        self._optimized_writer: Optional[OptimizedVectorStoreWriter] = None

    def close(self):
        """Clean up resources."""
        if self._optimized_writer:
            self._optimized_writer.close()

    async def write_documents(
        self,
        documents: List[Document],
        progress_callback: Optional[callable] = None,
        checkpoint_callback: Optional[callable] = None,
    ) -> None:
        """Write documents to vector store with parallel embedding.

        Args:
            documents: Documents to write.
            progress_callback: Optional progress update callback.
            checkpoint_callback: Optional checkpoint callback.
        """
        if not documents:
            return

        try:
            await self._write_parallel(
                documents, progress_callback, checkpoint_callback
            )
        except Exception as e:
            logger.warning(
                f"Parallel storage failed, falling back to standard: {e}"
            )
            await self._write_with_retry(documents)

    async def _write_parallel(
        self,
        documents: List[Document],
        progress_callback: Optional[callable] = None,
        checkpoint_callback: Optional[callable] = None,
    ) -> None:
        """Write documents using parallel embedding generation."""
        if not self._optimized_writer:
            self._optimized_writer = OptimizedVectorStoreWriter(
                self._vector_store.vector_store,
                self._vector_store.embeddings,
                progress_tracker=None,
                cache_service=self._embedding_cache,
            )

        if checkpoint_callback:
            self._optimized_writer.set_checkpoint_callback(checkpoint_callback)

        await self._optimized_writer.add_documents_optimized(
            documents,
            batch_size=settings.vector_store_batch_size,
            embedding_batch_size=settings.embedding_batch_size,
            max_concurrent_embeddings=settings.max_concurrent_embeddings,
        )

    async def _write_with_retry(self, documents: List[Document]) -> None:
        """Write documents with retry on failure."""

        async def store_documents():
            try:
                await self._vector_store.add_documents(documents)
            except Exception as e:
                raise StorageError(
                    f"Failed to store documents: {e}",
                    operation="add_documents",
                )

        await self._retry_manager.execute_with_retry_async(store_documents)

    async def write_table_documents(
        self,
        table_docs: List[Document],
        table_retriever: Any,
    ) -> bool:
        """Write table documents using multi-vector approach.

        Args:
            table_docs: Table documents to write.
            table_retriever: The table multi-vector retriever.

        Returns:
            True if successful, False if fallback needed.
        """
        if not table_docs or not table_retriever:
            return False

        langchain_docs = []
        for doc in table_docs:
            langchain_doc = LangchainDocument(
                page_content=doc.content,
                metadata=(
                    doc.metadata.model_dump()
                    if hasattr(doc.metadata, "model_dump")
                    else doc.metadata
                ),
            )
            langchain_docs.append(langchain_doc)

        try:
            await table_retriever.add_tables(langchain_docs)
            logger.info(
                f"Added {len(langchain_docs)} tables to multi-vector retriever"
            )
            return True
        except Exception as e:
            logger.warning(
                f"TABLE_FALLBACK: Failed to add {len(langchain_docs)} tables: {e}"
            )
            return False

    async def refresh_bm25_corpus(self) -> None:
        """Refresh BM25 corpus cache after ingestion."""
        try:
            self._vector_store.get_all_documents(refresh=True)
            logger.info("BM25 corpus cache refreshed after ingestion")
        except Exception as e:
            logger.warning(f"Failed to refresh BM25 corpus cache: {e}")
