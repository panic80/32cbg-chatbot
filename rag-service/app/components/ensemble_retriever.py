"""
Ensemble Retriever that combines multiple retrieval strategies with weighted scoring.

This provides a generic, LangChain-conformant solution for improving retrieval.
Migrated to extend LangChain's EnsembleRetriever with custom content boosting.
"""

from typing import List, Optional, Dict, Any, Tuple
import logging
import re
from collections import defaultdict
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_classic.retrievers import EnsembleRetriever as LangChainEnsembleRetriever
from pydantic import Field, validator

from app.utils.retry import with_retry_async
from app.components.base import BaseComponent

logger = logging.getLogger(__name__)


class WeightedEnsembleRetriever(LangChainEnsembleRetriever):
    """
    Enhanced ensemble retriever that extends LangChain's EnsembleRetriever
    with additional features:
    - Content-based boosting with regex patterns
    - Custom reranking strategies
    - Async support with retry logic
    
    This maintains compatibility with LangChain while adding domain-specific
    improvements for travel instruction queries.
    """
    
    boost_patterns: Dict[str, float] = Field(
        default_factory=lambda: {
            r"\$\d+\.?\d*": 0.3,  # Boost documents with dollar amounts
            r"\d{1,2}:\d{2}": 0.2,  # Boost documents with times
            r"\b\d+%\b": 0.2,  # Boost documents with percentages
        },
        description="Regex patterns and their boost scores"
    )
    
    rerank_strategy: str = Field(
        default="weighted_fusion",
        description="Strategy for combining results: weighted_fusion, content_boosted"
    )
    
    k: int = Field(
        default=5,
        description="Number of documents to return"
    )
    
    def __init__(self, **kwargs):
        """Initialize with proper parent class setup."""
        # Extract our custom fields before passing to parent
        boost_patterns = kwargs.pop("boost_patterns", None)
        rerank_strategy = kwargs.pop("rerank_strategy", "weighted_fusion")
        k = kwargs.pop("k", 5)
        
        # Initialize parent class
        super().__init__(**kwargs)
        
        # Set our custom attributes
        if boost_patterns is not None:
            self.boost_patterns = boost_patterns
        self.rerank_strategy = rerank_strategy
        self.k = k
    
    def _apply_content_boost(self, doc: Document) -> float:
        """Apply content-based boosting to document score."""
        boost = 0.0
        content = doc.page_content.lower()
        
        for pattern, boost_value in self.boost_patterns.items():
            if re.search(pattern, content, re.IGNORECASE):
                boost += boost_value
                logger.debug(f"Applied boost {boost_value} for pattern {pattern}")
        
        return boost
    
    def _rerank_with_content_boost(self, docs: List[Document]) -> List[Document]:
        """Apply content boosting to rerank documents."""
        doc_scores = []
        
        for rank, doc in enumerate(docs):
            # Base score from rank (inverse rank)
            base_score = 1.0 / (rank + 1)
            
            # Apply content boost
            content_boost = self._apply_content_boost(doc)
            
            # Combined score
            total_score = base_score + content_boost
            
            # Store score in metadata for transparency
            doc.metadata["ensemble_score"] = total_score
            doc.metadata["content_boost"] = content_boost
            
            doc_scores.append((doc, total_score))
        
        # Sort by score
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return just the documents
        return [doc for doc, _ in doc_scores]
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        """
        Get relevant documents using weighted ensemble approach.
        
        Extends parent class to add content boosting after fusion.
        """
        # Get fused results from parent class
        fused_docs = super()._get_relevant_documents(query, run_manager=run_manager)
        
        # Apply additional reranking if configured
        if self.rerank_strategy == "content_boosted":
            fused_docs = self._rerank_with_content_boost(fused_docs)
        
        # Limit to k documents
        return fused_docs[:self.k]
    
    @with_retry_async()
    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        """
        Async version with retry logic.
        
        Extends parent class to add content boosting after fusion.
        """
        # Get fused results from parent class
        fused_docs = await super()._aget_relevant_documents(query, run_manager=run_manager)
        
        # Apply additional reranking if configured
        if self.rerank_strategy == "content_boosted":
            fused_docs = self._rerank_with_content_boost(fused_docs)
        
        # Limit to k documents
        return fused_docs[:self.k]


class ContentBoostedEnsembleRetriever(BaseComponent, WeightedEnsembleRetriever):
    """
    Production-ready ensemble retriever with monitoring and caching.
    
    Combines BaseComponent capabilities with WeightedEnsembleRetriever
    for a fully-featured retriever with:
    - Performance monitoring
    - Error tracking
    - Caching support
    - Content boosting
    """
    
    def __init__(self, **kwargs):
        """Initialize both parent classes."""
        # Initialize BaseComponent
        BaseComponent.__init__(self, component_type="ensemble_retriever")
        
        # Initialize WeightedEnsembleRetriever
        WeightedEnsembleRetriever.__init__(self, **kwargs)
    
    @BaseComponent.monitor_performance
    async def retrieve(
        self,
        query: str,
        k: Optional[int] = None
    ) -> List[Document]:
        """
        Main retrieval method with monitoring.
        
        Args:
            query: Search query
            k: Number of documents to return (overrides default)
            
        Returns:
            List of relevant documents
        """
        # Override k if provided
        if k is not None:
            original_k = self.k
            self.k = k
        
        try:
            # Use parent's async retrieval
            docs = await self._aget_relevant_documents(query)
            
            # Log retrieval metrics
            self.log_event("retrieval_completed", {
                "query": query,
                "num_docs": len(docs),
                "num_retrievers": len(self.retrievers),
                "rerank_strategy": self.rerank_strategy
            })
            
            return docs
            
        finally:
            # Restore original k if overridden
            if k is not None:
                self.k = original_k