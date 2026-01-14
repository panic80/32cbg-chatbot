"""Vector store management for RAG service."""

import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

from langchain_core.documents import Document as LangchainDocument
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever
from langchain_community.vectorstores.utils import filter_complex_metadata

from app.core.config import settings
from app.core.logging import get_logger
from app.models.documents import Document, DocumentMetadata

logger = get_logger(__name__)


class VectorStoreManager:
    """Manages vector store operations."""
    
    def __init__(self):
        """Initialize vector store manager."""
        self.embeddings: Optional[Embeddings] = None
        self.vector_store: Optional[VectorStore] = None
        self.executor = ThreadPoolExecutor(max_workers=settings.parallel_embedding_workers)
        # Cache of all documents for BM25 corpus (populated on demand)
        self._all_documents_cache: Optional[List[LangchainDocument]] = None
        
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
        """Return all documents from the underlying vector store (cached).

        This is primarily used to provide a corpus for BM25 retrieval. For Chroma,
        we read documents directly from the underlying collection once and cache
        them for subsequent requests.

        Args:
            refresh: If True, forces reloading from the collection.

        Returns:
            List of LangChain Document objects representing the entire corpus.
        """
        try:
            if self._all_documents_cache is not None and not refresh:
                return self._all_documents_cache

            if hasattr(self.vector_store, "_collection") and self.vector_store._collection is not None:
                # Chroma collection access
                max_docs = getattr(settings, "bm25_max_corpus_docs", 0)
                get_kwargs = {"include": ["documents", "metadatas"]}
                if isinstance(max_docs, int) and max_docs > 0:
                    get_kwargs["limit"] = max_docs
                try:
                    results = self.vector_store._collection.get(**get_kwargs) or {}
                except TypeError as e:
                    if "limit" in get_kwargs:
                        logger.warning(
                            "Vector store get() does not support limit; falling back to full fetch"
                        )
                        get_kwargs.pop("limit", None)
                        results = self.vector_store._collection.get(**get_kwargs) or {}
                    else:
                        raise e
                documents = results.get("documents") or []
                metadatas = results.get("metadatas") or []

                langchain_docs: List[LangchainDocument] = []
                for i, content in enumerate(documents):
                    metadata = metadatas[i] if i < len(metadatas) and metadatas else {}
                    langchain_docs.append(LangchainDocument(page_content=content or "", metadata=metadata or {}))

                self._all_documents_cache = langchain_docs
                if isinstance(max_docs, int) and max_docs > 0 and len(langchain_docs) >= max_docs:
                    total_docs = None
                    if hasattr(self.vector_store._collection, "count"):
                        try:
                            total_docs = self.vector_store._collection.count()
                        except Exception:
                            total_docs = None
                    if total_docs and total_docs > max_docs:
                        logger.warning(
                            f"BM25 corpus capped at {max_docs} of {total_docs} documents"
                        )
                    else:
                        logger.warning(f"BM25 corpus capped at {max_docs} documents")
                logger.info(f"Loaded {len(langchain_docs)} documents for BM25 corpus cache")
                return langchain_docs

            # Fallback: no direct collection API available
            logger.warning("Vector store does not expose a collection; BM25 corpus unavailable")
            self._all_documents_cache = []
            return self._all_documents_cache

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
            
    def _create_vector_store(self) -> VectorStore:
        """Create vector store instance."""
        if settings.vector_store_type == "chroma":
            # Ensure persist directory exists
            os.makedirs(settings.chroma_persist_directory, exist_ok=True)
            
            return Chroma(
                collection_name=settings.chroma_collection_name,
                embedding_function=self.embeddings,
                persist_directory=settings.chroma_persist_directory,
                collection_metadata={"hnsw:space": "cosine"},
            )
        else:
            raise ValueError(f"Unsupported vector store type: {settings.vector_store_type}")
            
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
                ids.append(doc.id) # Use the document ID if available? The original code didn't set IDs explicitly in add_documents but LangChain usually generates them or uses what's in doc.id if it's a Document object? 
                # Wait, original code: langchain_doc = LangchainDocument(...); batch.append(langchain_doc); vector_store.add_documents(batch)
                # Chroma add_documents uses UUIDs if not provided.
                # Here we want to be consistent. Let's use doc.id if it exists, or let LangChain generate them if not. 
                # But to use add_texts with embeddings we need parallel lists.
                
            # Add documents in batches
            all_ids = []
            actual_batch_size = batch_size or settings.vector_store_batch_size
            
            for i in range(0, len(texts), actual_batch_size):
                batch_texts = texts[i:i + actual_batch_size]
                batch_metadatas = metadatas[i:i + actual_batch_size]
                # batch_ids = ids[i:i + actual_batch_size] # Let's omit IDs for now unless we are sure, or just rely on add_texts returning IDs.
                
                batch_embeddings = None
                if embeddings:
                    batch_embeddings = embeddings[i:i + actual_batch_size]
                
                # Run in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                
                if batch_embeddings:
                     batch_ids = await loop.run_in_executor(
                        self.executor,
                        lambda: self.vector_store.add_texts(
                            texts=batch_texts,
                            metadatas=batch_metadatas,
                            embeddings=batch_embeddings
                        )
                    )
                else:
                    # Fallback to creating Documents and using add_documents if no embeddings (or just use add_texts without embeddings)
                    # Let's use add_texts without embeddings for consistency
                     batch_ids = await loop.run_in_executor(
                        self.executor,
                        lambda: self.vector_store.add_texts(
                            texts=batch_texts,
                            metadatas=batch_metadatas
                        )
                    )
                    
                all_ids.extend(batch_ids)
                
                logger.info(f"Added batch {i//actual_batch_size + 1}: {len(batch_texts)} documents")
                
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
        """Get vector store statistics."""
        try:
            # Get collection info for Chroma
            if hasattr(self.vector_store, '_collection'):
                count = self.vector_store._collection.count()
                return {
                    "type": settings.vector_store_type,
                    "collection": settings.chroma_collection_name,
                    "document_count": count,
                    "persist_directory": settings.chroma_persist_directory,
                }
            else:
                return {
                    "type": settings.vector_store_type,
                    "status": "operational",
                }
                
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {
                "type": settings.vector_store_type,
                "status": "error",
                "error": str(e),
            }
            
    async def close(self) -> None:
        """Close vector store connections."""
        try:
            # Persist Chroma data
            if hasattr(self.vector_store, 'persist'):
                self.vector_store.persist()
                
            # Shutdown executor
            self.executor.shutdown(wait=True)
            logger.info("Vector store closed")
            
        except Exception as e:
            logger.error(f"Error closing vector store: {e}")
