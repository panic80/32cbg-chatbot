"""Vector store management for RAG service using PostgreSQL/pgvector."""

import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

from langchain_core.documents import Document as LangchainDocument
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGEngine, PGVectorStore
from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever
from sqlalchemy import create_engine, text
from sqlalchemy.exc import ProgrammingError

from app.core.config import settings
from app.core.logging import get_logger
from app.models.documents import Document, DocumentMetadata

logger = get_logger(__name__)


class VectorStoreManager:
    """Manages vector store operations with PostgreSQL/pgvector."""

    def __init__(self):
        """Initialize vector store manager."""
        self.embeddings: Optional[Embeddings] = None
        self.vector_store: Optional[PGVectorStore] = None
        self.executor = ThreadPoolExecutor(max_workers=settings.parallel_embedding_workers)
        # Cache of all documents for BM25 corpus (populated on demand)
        self._all_documents_cache: Optional[List[LangchainDocument]] = None
        # SQLAlchemy engine for direct SQL access (BM25 corpus, stats)
        self._sync_engine = None
        # PGEngine for langchain-postgres connection pool
        self._pg_engine: Optional[PGEngine] = None

    def _build_connection_string(self, async_driver: bool = False) -> str:
        """Build PostgreSQL connection string from settings.

        Args:
            async_driver: If True, use asyncpg driver; otherwise use psycopg
        """
        if settings.database_url:
            url = settings.database_url
            # Ensure correct driver based on async_driver flag
            if async_driver:
                if "+psycopg" in url:
                    url = url.replace("+psycopg", "+asyncpg")
                elif "postgresql://" in url and "+asyncpg" not in url:
                    url = url.replace("postgresql://", "postgresql+asyncpg://")
            else:
                if "+asyncpg" in url:
                    url = url.replace("+asyncpg", "+psycopg")
                elif "postgresql://" in url and "+psycopg" not in url:
                    url = url.replace("postgresql://", "postgresql+psycopg://")
            return url

        password = settings.postgres_password or ""
        driver = "asyncpg" if async_driver else "psycopg"
        return f"postgresql+{driver}://{settings.postgres_user}:{password}@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"

    async def initialize(self) -> None:
        """Initialize embeddings and vector store."""
        try:
            # Initialize embeddings
            self.embeddings = self._create_embeddings()
            logger.info("Embeddings initialized")

            # Initialize vector store
            self.vector_store = self._create_vector_store()
            logger.info(f"Vector store initialized: {settings.vector_store_type}")

        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise

    def get_all_documents(self, refresh: bool = False) -> List[LangchainDocument]:
        """Return all documents from PostgreSQL for BM25 corpus.

        PGVectorStore creates a table with columns:
        - id (UUID)
        - content (TEXT) - the page_content
        - embedding (VECTOR)
        - cmetadata (JSONB) - Note: langchain-postgres uses 'cmetadata' not 'metadata'

        Args:
            refresh: If True, forces reloading from the database.

        Returns:
            List of LangChain Document objects representing the entire corpus.
        """
        try:
            if self._all_documents_cache is not None and not refresh:
                return self._all_documents_cache

            if self._sync_engine is None:
                logger.warning("Sync engine not initialized; BM25 corpus unavailable")
                self._all_documents_cache = []
                return self._all_documents_cache

            max_docs = getattr(settings, "bm25_max_corpus_docs", 0)
            limit_clause = f"LIMIT {max_docs}" if max_docs > 0 else ""

            # Query the table created by PGVectorStore
            # langchain-postgres v2 uses: content (TEXT), langchain_metadata (JSON)
            query = text(f"""
                SELECT content, langchain_metadata
                FROM {settings.pgvector_table_name}
                {limit_clause}
            """)

            with self._sync_engine.connect() as conn:
                result = conn.execute(query)

                langchain_docs = []
                for row in result:
                    langchain_docs.append(
                        LangchainDocument(
                            page_content=row.content or "",
                            metadata=row.langchain_metadata or {}
                        )
                    )

                self._all_documents_cache = langchain_docs

                # Log warning if capped
                if max_docs > 0 and len(langchain_docs) >= max_docs:
                    count_result = conn.execute(
                        text(f"SELECT COUNT(*) FROM {settings.pgvector_table_name}")
                    )
                    total = count_result.scalar()
                    if total and total > max_docs:
                        logger.warning(f"BM25 corpus capped at {max_docs} of {total} documents")

                logger.info(f"Loaded {len(langchain_docs)} documents for BM25 corpus cache")
                return langchain_docs

        except Exception as e:
            logger.error(f"Failed to load all documents for BM25 corpus: {e}")
            self._all_documents_cache = []
            return self._all_documents_cache

    def _create_embeddings(self) -> Embeddings:
        """Create embeddings instance based on configuration."""
        if settings.openai_api_key:
            logger.info(f"Using OpenAI embeddings: {settings.openai_embedding_model} with {settings.openai_embedding_dimensions} dimensions")
            return OpenAIEmbeddings(
                api_key=settings.openai_api_key,
                model=settings.openai_embedding_model,
                dimensions=settings.openai_embedding_dimensions,
            )
        elif settings.google_api_key:
            logger.info(f"Using Google embeddings: {settings.google_embedding_model}")
            return GoogleGenerativeAIEmbeddings(
                google_api_key=settings.google_api_key,
                model=settings.google_embedding_model,
            )
        else:
            raise ValueError("No embedding API key configured")

    def _create_vector_store(self) -> PGVectorStore:
        """Create PGVectorStore instance using langchain-postgres factory method."""
        connection_string = self._build_connection_string(async_driver=False)

        # Store sync engine for direct SQL access (needed for BM25 corpus, stats)
        self._sync_engine = create_engine(connection_string)

        # Ensure the pgvector extension exists
        self._ensure_pgvector_extension()

        try:
            # Create PGEngine (manages connection pool for langchain-postgres)
            self._pg_engine = PGEngine.from_connection_string(url=connection_string)

            # Initialize vectorstore table if not exists
            try:
                self._pg_engine.init_vectorstore_table(
                    table_name=settings.pgvector_table_name,
                    vector_size=settings.pgvector_vector_dimensions,
                )
                logger.info(f"Created vectorstore table: {settings.pgvector_table_name}")
            except ProgrammingError:
                # Table already exists - this is fine
                logger.info(f"Vectorstore table already exists: {settings.pgvector_table_name}")

            # Create PGVectorStore using the factory method
            vector_store = PGVectorStore.create_sync(
                engine=self._pg_engine,
                table_name=settings.pgvector_table_name,
                embedding_service=self.embeddings,
            )
            logger.info(f"PGVectorStore initialized with table: {settings.pgvector_table_name}")
            return vector_store
        except Exception as e:
            logger.error(f"Failed to create PGVectorStore: {e}")
            raise

    def _ensure_pgvector_extension(self) -> None:
        """Ensure the pgvector extension is enabled in the database."""
        try:
            with self._sync_engine.connect() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.commit()
                logger.info("pgvector extension ensured")
        except Exception as e:
            logger.warning(f"Could not ensure pgvector extension (may already exist): {e}")

    async def add_documents(
        self,
        documents: List[Document],
        batch_size: Optional[int] = None,
        embeddings: Optional[List[List[float]]] = None
    ) -> List[str]:
        """Add documents to vector store."""
        try:
            # Convert to LangChain documents and prepare lists for add_texts
            texts = []
            metadatas = []
            ids = []

            for doc in documents:
                metadata = doc.metadata.model_dump() if hasattr(doc.metadata, 'model_dump') else doc.metadata
                metadata["id"] = doc.id
                metadata["created_at"] = doc.created_at.isoformat()

                # Filter out complex metadata types (lists, dicts, etc.)
                filtered_metadata = {}
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        filtered_metadata[key] = value
                    elif isinstance(value, list) and key == "tags":
                        # Convert tags list to comma-separated string
                        filtered_metadata[key] = ", ".join(str(v) for v in value)
                    else:
                        # Skip complex types or convert to string
                        logger.debug(f"Skipping complex metadata field: {key}")

                texts.append(doc.content)
                metadatas.append(filtered_metadata)
                ids.append(doc.id)

            # Add documents in batches
            all_ids = []
            actual_batch_size = batch_size or settings.vector_store_batch_size

            for i in range(0, len(texts), actual_batch_size):
                batch_texts = texts[i:i + actual_batch_size]
                batch_metadatas = metadatas[i:i + actual_batch_size]
                batch_ids = ids[i:i + actual_batch_size]

                batch_embeddings = None
                if embeddings:
                    batch_embeddings = embeddings[i:i + actual_batch_size]

                # Run in thread pool to avoid blocking
                loop = asyncio.get_event_loop()

                if batch_embeddings:
                    # Use add_embeddings when pre-computed embeddings exist
                    # Using default args in lambda to fix closure capture issues in the loop
                    # Pass ids=None to let PGVector generate proper UUIDs (original doc id is in metadata)
                    result_ids = await loop.run_in_executor(
                        self.executor,
                        lambda bt=batch_texts, be=batch_embeddings, bm=batch_metadatas:
                            self.vector_store.add_embeddings(
                                texts=bt,
                                embeddings=be,
                                metadatas=bm,
                                ids=None,  # Let PGVector generate proper UUIDs
                            )
                    )
                else:
                    # No embeddings - let add_texts generate them
                    # Pass ids=None to let PGVector generate proper UUIDs (original doc id is in metadata)
                    result_ids = await loop.run_in_executor(
                        self.executor,
                        lambda bt=batch_texts, bm=batch_metadatas:
                            self.vector_store.add_texts(
                                texts=bt,
                                metadatas=bm,
                                ids=None,  # Let PGVector generate proper UUIDs
                            )
                    )

                all_ids.extend(result_ids)

                logger.info(f"Added batch {i//actual_batch_size + 1}: {len(batch_texts)} documents")

            # Invalidate BM25 cache after adding documents
            self._all_documents_cache = None

            return all_ids

        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise

    async def search(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        search_type: str = "similarity"
    ) -> List[Tuple[LangchainDocument, float]]:
        """Search for similar documents."""
        try:
            loop = asyncio.get_event_loop()

            if search_type == "mmr":
                # Maximum Marginal Relevance search
                docs = await loop.run_in_executor(
                    self.executor,
                    lambda: self.vector_store.max_marginal_relevance_search_with_score(
                        query,
                        k=k,
                        filter=filter_dict,
                        fetch_k=settings.retrieval_fetch_k,
                        lambda_mult=settings.retrieval_lambda_mult,
                    )
                )
            else:
                # Similarity search
                docs = await loop.run_in_executor(
                    self.executor,
                    lambda: self.vector_store.similarity_search_with_score(
                        query,
                        k=k,
                        filter=filter_dict
                    )
                )

            return docs

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise

    async def delete_documents(
        self,
        ids: Optional[List[str]] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Delete documents from vector store."""
        try:
            if ids:
                # Delete by IDs
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    self.executor,
                    self.vector_store.delete,
                    ids
                )
                logger.info(f"Deleted {len(ids)} documents")

                # Invalidate BM25 cache
                self._all_documents_cache = None
            elif filter_dict:
                # Delete by filter - implementation depends on vector store
                logger.warning("Delete by filter not implemented for all vector stores")

            return True

        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            return False

    def get_retriever(
        self,
        search_kwargs: Optional[Dict[str, Any]] = None
    ) -> BaseRetriever:
        """Get a retriever instance."""
        default_kwargs = {
            "k": settings.retrieval_k,
            "search_type": settings.retrieval_search_type,
        }

        if settings.retrieval_search_type == "mmr":
            default_kwargs.update({
                "fetch_k": settings.retrieval_fetch_k,
                "lambda_mult": settings.retrieval_lambda_mult,
            })

        if search_kwargs:
            default_kwargs.update(search_kwargs)

        return self.vector_store.as_retriever(**default_kwargs)

    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get vector store statistics from PostgreSQL."""
        try:
            if self._sync_engine is None:
                return {
                    "type": settings.vector_store_type,
                    "status": "not initialized",
                }

            with self._sync_engine.connect() as conn:
                result = conn.execute(
                    text(f"SELECT COUNT(*) FROM {settings.pgvector_table_name}")
                )
                count = result.scalar()

            return {
                "type": settings.vector_store_type,
                "table": settings.pgvector_table_name,
                "document_count": count,
                "database": settings.postgres_db,
                "host": settings.postgres_host,
                "vector_dimensions": settings.pgvector_vector_dimensions,
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"type": settings.vector_store_type, "status": "error", "error": str(e)}

    async def close(self) -> None:
        """Close vector store connections."""
        try:
            # Close PGEngine connection pool (async method)
            if self._pg_engine:
                try:
                    await self._pg_engine.close()
                    logger.info("PGEngine closed")
                except Exception as e:
                    logger.warning(f"Error closing PGEngine: {e}")

            # Dispose SQLAlchemy engine
            if self._sync_engine:
                self._sync_engine.dispose()
                logger.info("SQLAlchemy engine disposed")

            # Shutdown executor
            self.executor.shutdown(wait=True)
            logger.info("Vector store closed")

        except Exception as e:
            logger.error(f"Error closing vector store: {e}")
