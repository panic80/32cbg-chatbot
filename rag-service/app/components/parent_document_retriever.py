"""
Parent Document Retriever component for hierarchical retrieval.

This module provides a retriever that returns parent documents
when child chunks match, preserving full context.
"""

from typing import List, Dict, Any, Optional, Set
import logging
import hashlib

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.vectorstores import VectorStore
from pydantic import Field

from app.components.base import BaseComponent
from app.core.logging import get_logger

logger = get_logger(__name__)


class ParentDocumentIndex:
    """Index mapping child chunks to parent documents."""
    
    def __init__(self):
        self.child_to_parent: Dict[str, str] = {}
        self.parent_docs: Dict[str, Document] = {}
        self.parent_to_children: Dict[str, List[str]] = {}
    
    def add_documents(self, parent_doc: Document, child_docs: List[Document]):
        """
        Add a parent document and its children to the index.
        
        Args:
            parent_doc: Parent document
            child_docs: List of child chunks
        """
        # Generate parent ID
        parent_id = self._generate_id(parent_doc)
        
        # Store parent document
        self.parent_docs[parent_id] = parent_doc
        
        # Map children to parent
        child_ids = []
        for child in child_docs:
            child_id = self._generate_id(child)
            self.child_to_parent[child_id] = parent_id
            child_ids.append(child_id)
        
        # Store parent to children mapping
        self.parent_to_children[parent_id] = child_ids
    
    def get_parent(self, child_doc: Document) -> Optional[Document]:
        """
        Get parent document for a child chunk.
        
        Args:
            child_doc: Child document
            
        Returns:
            Parent document if found
        """
        child_id = self._generate_id(child_doc)
        parent_id = self.child_to_parent.get(child_id)
        
        if parent_id:
            return self.parent_docs.get(parent_id)
        
        return None
    
    def get_parent_by_id(self, child_id: str) -> Optional[Document]:
        """
        Get parent document by child ID.
        
        Args:
            child_id: Child document ID
            
        Returns:
            Parent document if found
        """
        parent_id = self.child_to_parent.get(child_id)
        
        if parent_id:
            return self.parent_docs.get(parent_id)
        
        return None
    
    def _generate_id(self, doc: Document) -> str:
        """
        Generate unique ID for a document.
        
        Args:
            doc: Document to generate ID for
            
        Returns:
            Document ID
        """
        # Use content hash + source for uniqueness
        content = doc.page_content
        source = doc.metadata.get("source", "")
        
        hash_input = f"{source}:{content}"
        return hashlib.md5(hash_input.encode()).hexdigest()


class TravelParentDocumentRetriever(BaseRetriever, BaseComponent):
    """
    Parent document retriever for travel documents.
    
    Returns full parent documents when child chunks match queries,
    providing complete context for travel instructions.
    """
    
    child_retriever: BaseRetriever = Field(description="Retriever for child chunks")
    parent_index: ParentDocumentIndex = Field(description="Parent document index")
    k: int = Field(default=5, description="Number of parent documents to return")
    score_threshold: float = Field(default=0.0, description="Minimum score threshold")
    
    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True
    
    def __init__(
        self,
        child_retriever: BaseRetriever,
        parent_documents: Optional[List[Document]] = None,
        k: int = 5,
        score_threshold: float = 0.0,
        **kwargs
    ):
        """
        Initialize the parent document retriever.
        
        Args:
            child_retriever: Retriever for finding relevant child chunks
            parent_documents: Optional list of parent documents to index
            k: Number of parent documents to return
            score_threshold: Minimum relevance score
        """
        # Initialize BaseComponent
        BaseComponent.__init__(self, component_type="retriever", component_name="parent_document")
        
        # Create parent index
        parent_index = ParentDocumentIndex()
        
        # Initialize BaseRetriever with fields
        super().__init__(
            child_retriever=child_retriever,
            parent_index=parent_index,
            k=k,
            score_threshold=score_threshold,
            **kwargs
        )
        
        # Index parent documents if provided
        if parent_documents:
            self._index_documents(parent_documents)
        
        logger.info(f"Initialized parent document retriever with k={k}")
    
    def _index_documents(self, documents: List[Document]):
        """
        Index parent documents with their chunks.
        
        Args:
            documents: Parent documents to index
        """
        for doc in documents:
            # Split parent into chunks (simplified - in practice use proper splitter)
            chunks = self._split_document(doc)
            
            # Add to index
            self.parent_index.add_documents(doc, chunks)
        
        logger.info(f"Indexed {len(documents)} parent documents")
    
    def _split_document(self, document: Document) -> List[Document]:
        """
        Split a document into chunks.
        
        Args:
            document: Document to split
            
        Returns:
            List of chunks
        """
        # Simplified chunking - in practice use proper text splitter
        content = document.page_content
        chunk_size = 1000
        overlap = 200
        
        chunks = []
        start = 0
        
        while start < len(content):
            end = min(start + chunk_size, len(content))
            
            # Find natural break point
            if end < len(content):
                # Look for sentence end
                for sep in ['. ', '.\n', '? ', '! ']:
                    pos = content.rfind(sep, start, end)
                    if pos > start:
                        end = pos + len(sep)
                        break
            
            chunk_content = content[start:end]
            
            # Create chunk document
            chunk = Document(
                page_content=chunk_content,
                metadata={
                    **document.metadata,
                    "chunk_index": len(chunks),
                    "parent_source": document.metadata.get("source", ""),
                    "is_chunk": True
                }
            )
            
            chunks.append(chunk)
            
            # Move start position
            start = end - overlap if end < len(content) else end
        
        return chunks
    
    def add_parent_documents(self, documents: List[Document]):
        """
        Add new parent documents to the index.
        
        Args:
            documents: Parent documents to add
        """
        self._index_documents(documents)
        
        # Also add chunks to child retriever if it supports it
        if hasattr(self.child_retriever, 'add_documents'):
            all_chunks = []
            for doc in documents:
                chunks = self._split_document(doc)
                all_chunks.extend(chunks)
            
            self.child_retriever.add_documents(all_chunks)
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        """
        Get relevant parent documents based on child chunk matches.
        
        Args:
            query: Search query
            run_manager: Callback manager
            
        Returns:
            List of parent documents
        """
        # Get relevant child chunks
        child_docs = self.child_retriever.get_relevant_documents(
            query,
            callbacks=run_manager.get_child() if run_manager else None
        )
        
        # Track unique parent documents
        seen_parents: Set[str] = set()
        parent_docs = []
        
        # Get parent documents for matching chunks
        for child in child_docs:
            # Try to get parent
            parent = self.parent_index.get_parent(child)
            
            if parent:
                parent_id = self.parent_index._generate_id(parent)
                
                # Avoid duplicates
                if parent_id not in seen_parents:
                    seen_parents.add(parent_id)
                    
                    # Add relevance score from child if available
                    if hasattr(child, 'metadata') and 'score' in child.metadata:
                        parent.metadata['relevance_score'] = child.metadata['score']
                    
                    parent_docs.append(parent)
                    
                    # Stop if we have enough
                    if len(parent_docs) >= self.k:
                        break
            else:
                # If no parent found, use the child itself
                logger.debug(f"No parent found for chunk: {child.page_content[:100]}...")
                
                # Check if this might be a full document already
                if not child.metadata.get("is_chunk", False):
                    doc_id = self.parent_index._generate_id(child)
                    if doc_id not in seen_parents:
                        seen_parents.add(doc_id)
                        parent_docs.append(child)
        
        # Log retrieval
        self._log_event("retrieve", {
            "query": query,
            "num_child_matches": len(child_docs),
            "num_parent_docs": len(parent_docs),
            "method": "parent_document"
        })
        
        return parent_docs[:self.k]
    
    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        """
        Async get relevant parent documents.
        
        Args:
            query: Search query
            run_manager: Callback manager
            
        Returns:
            List of parent documents
        """
        # Get relevant child chunks
        child_docs = await self.child_retriever.aget_relevant_documents(
            query,
            callbacks=run_manager.get_child() if run_manager else None
        )
        
        # Track unique parent documents
        seen_parents: Set[str] = set()
        parent_docs = []
        
        # Get parent documents for matching chunks
        for child in child_docs:
            # Try to get parent
            parent = self.parent_index.get_parent(child)
            
            if parent:
                parent_id = self.parent_index._generate_id(parent)
                
                # Avoid duplicates
                if parent_id not in seen_parents:
                    seen_parents.add(parent_id)
                    
                    # Add relevance score from child if available
                    if hasattr(child, 'metadata') and 'score' in child.metadata:
                        parent.metadata['relevance_score'] = child.metadata['score']
                    
                    parent_docs.append(parent)
                    
                    # Stop if we have enough
                    if len(parent_docs) >= self.k:
                        break
            else:
                # If no parent found, use the child itself
                logger.debug(f"No parent found for chunk: {child.page_content[:100]}...")
                
                # Check if this might be a full document already
                if not child.metadata.get("is_chunk", False):
                    doc_id = self.parent_index._generate_id(child)
                    if doc_id not in seen_parents:
                        seen_parents.add(doc_id)
                        parent_docs.append(child)
        
        # Log retrieval
        self._log_event("retrieve", {
            "query": query,
            "num_child_matches": len(child_docs),
            "num_parent_docs": len(parent_docs),
            "method": "parent_document"
        })
        
        return parent_docs[:self.k]