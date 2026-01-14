"""
BM25 Retriever component for keyword-based retrieval.

This module provides a BM25 retriever that extends LangChain's BM25Retriever
with additional features for the travel domain.
"""

from typing import List, Dict, Any, Optional
import logging
import os

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_community.retrievers import BM25Retriever as LangChainBM25Retriever
from pydantic import Field

from app.components.base import BaseComponent
from app.core.logging import get_logger

logger = get_logger(__name__)


class TravelBM25Retriever(BaseRetriever, BaseComponent):
    """
    BM25 retriever optimized for travel documents.
    
    Extends LangChain's BM25Retriever with:
    - Travel-specific preprocessing
    - Performance monitoring
    - Async support
    """
    
    bm25_retriever: LangChainBM25Retriever = Field(description="Underlying BM25 retriever")
    k: int = Field(default=10, description="Number of documents to retrieve")
    preprocess_query: bool = Field(default=True, description="Whether to preprocess queries")
    component_type: str = Field(default="retriever", description="Component category identifier")
    component_name: str = Field(default="bm25", description="Component name for logging and metrics")
    
    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True
    
    def __init__(
        self,
        documents: Optional[List[Document]] = None,
        k: int = 10,
        preprocess_query: bool = True,
        component_name: str = "bm25",
        index_path: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the BM25 retriever.
        
        Args:
            documents: List of documents to index (optional if loading from disk)
            k: Number of documents to retrieve
            preprocess_query: Whether to preprocess queries
            index_path: Path to pickled BM25 index (optional)
        """
        bm25_retriever = None
        
        # Try loading from disk first if no documents provided or explicit path given
        if not documents:
            try:
                if not index_path:
                    # Determine base data directory
                    if os.path.exists("/app/data"):
                        base_data_dir = "/app/data"
                    else:
                        # Fallback to local data directory relative to project root
                        # .../app/components/bm25_retriever.py -> .../
                        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                        base_data_dir = os.path.join(project_root, "data")
                    
                    index_path = os.path.join(base_data_dir, "bm25", "bm25_retriever.pkl")
                
                if os.path.exists(index_path):
                    import pickle
                    logger.info(f"Loading BM25 index from {index_path}...")
                    with open(index_path, "rb") as f:
                        bm25_retriever = pickle.load(f)
                    
                    # Update k if different
                    bm25_retriever.k = k
                    logger.info(f"Successfully loaded BM25 index with {len(bm25_retriever.docs)} documents")
            except Exception as e:
                logger.warning(f"Failed to load BM25 index from disk: {e}")

        # Fallback to building from documents if load failed or documents explicitly provided
        if bm25_retriever is None:
            if documents:
                logger.info(f"Building BM25 index in-memory with {len(documents)} documents")
                bm25_retriever = LangChainBM25Retriever.from_documents(documents, k=k)
            else:
                # Initialize empty if absolutely nothing available (prevents crash, but won't retrieve)
                logger.warning("No documents or index provided for BM25. Initializing empty.")
                bm25_retriever = LangChainBM25Retriever.from_documents([Document(page_content="")], k=k)

        # Initialize BaseRetriever with fields
        super().__init__(
            bm25_retriever=bm25_retriever,
            k=k,
            preprocess_query=preprocess_query,
            component_type="retriever",
            component_name=component_name,
            **kwargs
        )

        # Initialize BaseComponent after BaseRetriever so attribute mutation is
        # compatible with pydantic's BaseModel internals.
        BaseComponent.__init__(self, component_type="retriever", component_name=component_name)
    
    def _preprocess_query(self, query: str) -> str:
        """
        Preprocess query for better BM25 matching.
        
        Args:
            query: Original query
            
        Returns:
            Preprocessed query
        """
        # Add travel-specific synonyms and expansions
        query_lower = query.lower()
        
        # Expand abbreviations
        abbreviations = {
            "pmv": "private motor vehicle",
            "td": "temporary duty",
            "tdy": "temporary duty",
            "cf": "canadian forces",
            "caf": "canadian armed forces",
            "gmt": "government motor transport",
            "per diem": "daily allowance",
            "km": "kilometer kilometre",
            "govt": "government",
            "accom": "accommodation",
            "trans": "transportation"
        }
        
        expanded_query = query
        for abbr, expansion in abbreviations.items():
            if abbr in query_lower:
                expanded_query += f" {expansion}"
        
        # Add keyword variations for common searches
        if "rate" in query_lower or "allowance" in query_lower:
            expanded_query += " table amount dollar per day daily"
        
        if "meal" in query_lower:
            expanded_query += " breakfast lunch dinner food"
        
        if "travel" in query_lower:
            expanded_query += " trip journey transportation"
        
        return expanded_query
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        """
        Get relevant documents using BM25.
        
        Args:
            query: Search query
            run_manager: Callback manager
            
        Returns:
            List of relevant documents
        """
        # Preprocess query if enabled
        if self.preprocess_query:
            processed_query = self._preprocess_query(query)
            logger.debug(f"Expanded query: '{query}' -> '{processed_query}'")
        else:
            processed_query = query
        
        # Get documents from underlying BM25 retriever
        docs = self.bm25_retriever.get_relevant_documents(
            processed_query,
            callbacks=run_manager.get_child() if run_manager else None
        )
        
        # Log retrieval
        self._log_event("retrieve", {
            "query": query,
            "processed_query": processed_query,
            "num_results": len(docs),
            "method": "bm25"
        })
        
        return docs[:self.k]
    
    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        """
        Async get relevant documents.
        
        BM25 is synchronous, so we just wrap the sync call.
        """
        import asyncio
        return await asyncio.to_thread(
            self._get_relevant_documents,
            query,
            run_manager=run_manager
        )
    
    def update_documents(self, documents: List[Document]):
        """
        Update the BM25 index with new documents.
        
        Args:
            documents: New list of documents
        """
        # Recreate the BM25 retriever with new documents
        self.bm25_retriever = LangChainBM25Retriever.from_documents(
            documents, 
            k=self.k
        )
        
        logger.info(f"Updated BM25 index with {len(documents)} documents")
        
        self._log_event("update_index", {
            "num_documents": len(documents)
        })
