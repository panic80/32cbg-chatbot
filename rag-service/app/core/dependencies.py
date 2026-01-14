"""Dependency injection and singleton management for core services."""

from typing import Optional
from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

from app.core.config import settings
from app.core.vectorstore import VectorStoreManager
from app.core.logging import get_logger

logger = get_logger(__name__)

# Global instances
_vector_store_manager: Optional[VectorStoreManager] = None
_embeddings: Optional[Embeddings] = None
_vectorstore: Optional[Chroma] = None


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


async def get_vectorstore() -> Chroma:
    """Get or create vector store instance."""
    global _vectorstore
    
    if _vectorstore is None:
        embeddings = get_embeddings()
        
        # Create Chroma instance
        import os
        os.makedirs(settings.chroma_persist_directory, exist_ok=True)
        
        _vectorstore = Chroma(
            collection_name=settings.chroma_collection_name,
            embedding_function=embeddings,
            persist_directory=settings.chroma_persist_directory,
            collection_metadata={"hnsw:space": "cosine"},
        )
        logger.info("Vector store instance created")
    
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