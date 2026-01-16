"""Dependency injection and singleton management for core services."""

from typing import Optional
from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVectorStore

from app.core.config import settings
from app.core.vectorstore import VectorStoreManager
from app.core.logging import get_logger

logger = get_logger(__name__)

# Global instances
_vector_store_manager: Optional[VectorStoreManager] = None
_embeddings: Optional[Embeddings] = None
_vectorstore: Optional[PGVectorStore] = None


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """Get or create embeddings instance."""
    global _embeddings

    if _embeddings is None:
        if settings.openai_api_key:
            logger.info(f"Creating OpenAI embeddings: {settings.openai_embedding_model}")
            _embeddings = OpenAIEmbeddings(
                api_key=settings.openai_api_key,
                model=settings.openai_embedding_model,
                dimensions=settings.openai_embedding_dimensions,
            )
        elif settings.google_api_key:
            logger.info(f"Creating Google embeddings: {settings.google_embedding_model}")
            _embeddings = GoogleGenerativeAIEmbeddings(
                google_api_key=settings.google_api_key,
                model=settings.google_embedding_model,
            )
        else:
            raise ValueError("No embedding API key configured")

    return _embeddings


async def get_vectorstore() -> PGVectorStore:
    """Get or create vector store instance."""
    global _vectorstore

    if _vectorstore is None:
        embeddings = get_embeddings()

        # Build connection string
        if settings.database_url:
            connection_string = settings.database_url
            # Ensure psycopg driver
            if "+asyncpg" in connection_string:
                connection_string = connection_string.replace("+asyncpg", "+psycopg")
            elif "postgresql://" in connection_string and "+psycopg" not in connection_string:
                connection_string = connection_string.replace("postgresql://", "postgresql+psycopg://")
        else:
            password = settings.postgres_password or ""
            connection_string = f"postgresql+psycopg://{settings.postgres_user}:{password}@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"

        _vectorstore = PGVectorStore(
            connection=connection_string,
            collection_name=settings.pgvector_table_name,
            embeddings=embeddings,
            use_jsonb=True,
        )
        logger.info("PGVectorStore instance created")

    return _vectorstore


async def get_vector_store_manager() -> VectorStoreManager:
    """Get or create vector store manager instance."""
    global _vector_store_manager

    if _vector_store_manager is None:
        _vector_store_manager = VectorStoreManager()
        await _vector_store_manager.initialize()
        logger.info("Vector store manager initialized")

    return _vector_store_manager


def set_vector_store_manager(manager: VectorStoreManager):
    """Set the global vector store manager instance."""
    global _vector_store_manager
    _vector_store_manager = manager


def reset_singletons():
    """Reset all singleton instances (useful for testing)."""
    global _vector_store_manager, _embeddings, _vectorstore
    _vector_store_manager = None
    _embeddings = None
    _vectorstore = None
    get_embeddings.cache_clear()
