"""
Authority Reranker component for boosting authoritative sources.

This module provides a reranker that boosts documents from
authoritative sources in the travel domain.
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
import re

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import Field

from app.components.base import BaseComponent
from app.core.logging import get_logger

logger = get_logger(__name__)


class AuthorityReranker(BaseComponent):
    """
    Reranker that boosts documents from authoritative sources.
    
    Specifically designed for Canadian Forces travel documents,
    prioritizing official sources and recent content.
    """
    
    # Authority score mappings
    AUTHORITY_SCORES = {
        # Primary sources (highest authority)
        "cftdti": 1.0,  # Canadian Forces TDI
        "njc": 0.9,     # National Joint Council
        "treasury": 0.85,  # Treasury Board
        "dnd": 0.8,     # Department of National Defence
        
        # Secondary sources
        "cafconnection": 0.6,
        "canada.ca": 0.5,
        "forces.gc.ca": 0.5,
        
        # Default
        "default": 0.3
    }
    
    def __init__(self, boost_factor: float = 2.0):
        """
        Initialize the authority reranker.
        
        Args:
            boost_factor: Factor to multiply authority scores by
        """
        super().__init__(component_type="reranker", component_name="authority")
        self.boost_factor = boost_factor
        logger.info(f"Initialized authority reranker with boost factor {boost_factor}")
    
    def rerank(
        self,
        documents: List[Document],
        query: Optional[str] = None
    ) -> List[Document]:
        """
        Rerank documents based on source authority.
        
        Args:
            documents: List of documents to rerank
            query: Optional query for context-aware reranking
            
        Returns:
            Reranked list of documents
        """
        # Score each document
        scored_docs = []
        for doc in documents:
            score = self._calculate_authority_score(doc, query)
            scored_docs.append((doc, score))
        
        # Sort by score (descending)
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Extract reranked documents
        reranked = [doc for doc, score in scored_docs]
        
        # Log reranking
        self._log_event("rerank", {
            "num_documents": len(documents),
            "query": query,
            "top_sources": [
                {
                    "source": doc.metadata.get("source", "unknown"),
                    "score": score
                }
                for doc, score in scored_docs[:5]
            ]
        })
        
        return reranked
    
    def _calculate_authority_score(
        self,
        document: Document,
        query: Optional[str] = None
    ) -> float:
        """
        Calculate authority score for a document.
        
        Args:
            document: Document to score
            query: Optional query for context-aware scoring
            
        Returns:
            Authority score
        """
        base_score = 1.0
        
        # Get source authority score
        source = document.metadata.get("source", "").lower()
        authority_score = self._get_source_authority(source)
        base_score *= (1 + authority_score * self.boost_factor)
        
        # Boost for official document types
        doc_type = document.metadata.get("document_type", "").lower()
        if doc_type in ["policy", "directive", "regulation", "official"]:
            base_score *= 1.5
        elif doc_type in ["guide", "handbook", "manual"]:
            base_score *= 1.3
        
        # Boost for recent content
        year = self._extract_year(document)
        if year:
            current_year = 2025  # Hardcoded for consistency
            age = current_year - year
            if age <= 1:
                base_score *= 1.4  # Very recent
            elif age <= 3:
                base_score *= 1.2  # Recent
            elif age >= 10:
                base_score *= 0.7  # Older content
        
        # Query-specific boosting
        if query:
            query_lower = query.lower()
            content_lower = document.page_content.lower()
            
            # Boost for title/header matches
            lines = content_lower.split('\n')
            if lines and query_lower in lines[0]:
                base_score *= 1.3
            
            # Boost for exact phrase matches
            if f'"{query_lower}"' in content_lower or f"'{query_lower}'" in content_lower:
                base_score *= 1.2
        
        # Boost for structured content (tables, lists)
        if self._has_structured_content(document):
            base_score *= 1.1
        
        return base_score
    
    def _get_source_authority(self, source: str) -> float:
        """
        Get authority score for a source.
        
        Args:
            source: Source URL or identifier
            
        Returns:
            Authority score
        """
        source_lower = source.lower()
        
        # Check each authority pattern
        for pattern, score in self.AUTHORITY_SCORES.items():
            if pattern in source_lower:
                return score
        
        # Check for government domains
        if any(gov in source_lower for gov in [".gc.ca", ".gov.ca", "government"]):
            return 0.4
        
        return self.AUTHORITY_SCORES["default"]
    
    def _extract_year(self, document: Document) -> Optional[int]:
        """
        Extract year from document content or metadata.
        
        Args:
            document: Document to extract year from
            
        Returns:
            Year if found, None otherwise
        """
        # Check metadata first
        if "year" in document.metadata:
            try:
                return int(document.metadata["year"])
            except (ValueError, TypeError):
                pass
        
        if "date" in document.metadata:
            date_str = str(document.metadata["date"])
            year_match = re.search(r'20\d{2}', date_str)
            if year_match:
                return int(year_match.group())
        
        # Extract from content
        content = document.page_content
        
        # Look for year patterns
        year_patterns = [
            r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+(20\d{2})',
            r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+(20\d{2})',
            r'(20\d{2})-\d{2}-\d{2}',  # ISO date
            r'\b(20\d{2})\b'  # Any 4-digit year starting with 20
        ]
        
        for pattern in year_patterns:
            match = re.search(pattern, content)
            if match:
                try:
                    year = int(match.group(1))
                    if 2000 <= year <= 2030:  # Reasonable range
                        return year
                except (ValueError, IndexError):
                    continue
        
        return None
    
    def _has_structured_content(self, document: Document) -> bool:
        """
        Check if document has structured content like tables or lists.
        
        Args:
            document: Document to check
            
        Returns:
            True if structured content is detected
        """
        content = document.page_content
        
        # Check for table indicators
        if any(indicator in content for indicator in ['|', '\t\t', '│', '─']):
            return True
        
        # Check for list patterns
        list_patterns = [
            r'^\s*[-•*]\s+',  # Bullet points
            r'^\s*\d+\.\s+',  # Numbered lists
            r'^\s*[a-z]\)\s+',  # Letter lists
        ]
        
        lines = content.split('\n')
        list_count = 0
        for line in lines:
            for pattern in list_patterns:
                if re.match(pattern, line):
                    list_count += 1
                    break
        
        # Consider it structured if multiple list items found
        return list_count >= 3


class AuthorityRerankingRetriever(BaseRetriever):
    """
    Retriever wrapper that applies authority reranking.
    """
    
    base_retriever: BaseRetriever = Field(description="Base retriever")
    reranker: AuthorityReranker = Field(description="Authority reranker")
    
    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True
    
    def __init__(
        self,
        base_retriever: BaseRetriever,
        boost_factor: float = 2.0,
        **kwargs
    ):
        """
        Initialize the authority reranking retriever.
        
        Args:
            base_retriever: Base retriever to wrap
            boost_factor: Authority boost factor
        """
        reranker = AuthorityReranker(boost_factor=boost_factor)
        super().__init__(
            base_retriever=base_retriever,
            reranker=reranker,
            **kwargs
        )
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        """
        Get documents from base retriever and rerank by authority.
        
        Args:
            query: Search query
            run_manager: Callback manager
            
        Returns:
            Reranked documents
        """
        # Get documents from base retriever
        docs = self.base_retriever.get_relevant_documents(
            query,
            callbacks=run_manager.get_child() if run_manager else None
        )
        
        # Rerank by authority
        reranked_docs = self.reranker.rerank(docs, query)
        
        return reranked_docs
    
    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        """
        Async get documents and rerank.
        
        Args:
            query: Search query
            run_manager: Callback manager
            
        Returns:
            Reranked documents
        """
        # Get documents from base retriever
        docs = await self.base_retriever.aget_relevant_documents(
            query,
            callbacks=run_manager.get_child() if run_manager else None
        )
        
        # Rerank by authority (sync operation)
        reranked_docs = self.reranker.rerank(docs, query)
        
        return reranked_docs