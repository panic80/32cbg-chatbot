"""
Contextual Compression Retriever for filtering and extracting relevant content.

This module implements LangChain's ContextualCompressionRetriever with multiple
compressors optimized for travel instruction queries.
"""

from typing import List, Optional, Dict, Any
import logging
from enum import Enum

from langchain_core.language_models import BaseLLM
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsFilter,
    LLMChainExtractor,
    LLMChainFilter
)
from langchain_core.prompts import PromptTemplate
from pydantic import Field, BaseModel

from app.components.base import BaseComponent

logger = logging.getLogger(__name__)


class CompressionMode(str, Enum):
    """Compression modes for different use cases."""
    EXTRACTIVE = "extractive"  # Extract only relevant portions
    FILTER = "filter"  # Filter out irrelevant documents
    HYBRID = "hybrid"  # Both filter and extract
    EMBEDDINGS_ONLY = "embeddings_only"  # Only use embeddings filter


# Prompts for different compression tasks
TRAVEL_EXTRACTION_PROMPT = PromptTemplate(
    template="""Given the following question and context, extract only the parts 
of the context that are relevant to answering the question. Focus on specific 
values, rates, policies, and procedures mentioned.

Question: {question}
Context: {context}

Relevant portions (if none, return "NO_RELEVANT_CONTENT"):""",
    input_variables=["question", "context"]
)

TRAVEL_FILTER_PROMPT = PromptTemplate(
    template="""Given the following question and context, determine if the context 
contains information relevant to answering the question about travel instructions,
allowances, or policies.

Question: {question}
Context: {context}

Is this context relevant? Answer only YES or NO:""",
    input_variables=["question", "context"]
)


class CompressionConfig(BaseModel):
    """Configuration for contextual compression."""
    mode: CompressionMode = Field(default=CompressionMode.EXTRACTIVE)
    similarity_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    min_chunk_length: int = Field(default=500, description="Minimum chunk length to compress")
    top_k: Optional[int] = Field(default=None, description="Limit results after compression")
    chunk_size: int = Field(default=1000, description="Size for chunking before compression")
    use_async: bool = Field(default=True)


class TravelContextualCompressor(BaseComponent):
    """
    Contextual compression optimized for travel instruction queries.
    
    Features:
    - Multiple compression modes
    - Query-specific thresholds
    - Performance monitoring
    - Fallback strategies
    """
    
    def __init__(
        self,
        base_retriever: BaseRetriever,
        llm: Optional[BaseLLM] = None,
        embeddings: Optional[Embeddings] = None,
        config: Optional[CompressionConfig] = None
    ):
        """
        Initialize the contextual compressor.
        
        Args:
            base_retriever: The retriever to compress results from
            llm: Language model for extraction/filtering
            embeddings: Embeddings for similarity filtering
            config: Compression configuration
        """
        super().__init__(component_type="contextual_compressor")
        
        self.base_retriever = base_retriever
        self.llm = llm
        self.embeddings = embeddings
        self.config = config or CompressionConfig()
        
        # Create compressor based on configuration
        self.compressor = self._create_compressor()
        
        # Create the contextual compression retriever
        self.retriever = ContextualCompressionRetriever(
            base_compressor=self.compressor,
            base_retriever=self.base_retriever
        )
    
    def _create_compressor(self):
        """Create the appropriate compressor based on configuration."""
        compressors = []
        
        if self.config.mode == CompressionMode.EMBEDDINGS_ONLY:
            if not self.embeddings:
                raise ValueError("Embeddings required for EMBEDDINGS_ONLY mode")
            
            return EmbeddingsFilter(
                embeddings=self.embeddings,
                similarity_threshold=self.config.similarity_threshold
            )
        
        # Add embeddings filter if available
        if self.embeddings:
            compressors.append(
                EmbeddingsFilter(
                    embeddings=self.embeddings,
                    similarity_threshold=self.config.similarity_threshold
                )
            )
        
        # Add LLM-based components if available
        if self.llm:
            if self.config.mode in [CompressionMode.FILTER, CompressionMode.HYBRID]:
                # Add filter to remove irrelevant documents
                compressors.append(
                    LLMChainFilter.from_llm(
                        llm=self.llm,
                        prompt=TRAVEL_FILTER_PROMPT
                    )
                )
            
            if self.config.mode in [CompressionMode.EXTRACTIVE, CompressionMode.HYBRID]:
                # Add extractor to get relevant portions
                compressors.append(
                    LLMChainExtractor.from_llm(
                        llm=self.llm,
                        prompt=TRAVEL_EXTRACTION_PROMPT
                    )
                )
        
        if not compressors:
            raise ValueError("No compressors available with current configuration")
        
        # Return single compressor or pipeline
        if len(compressors) == 1:
            return compressors[0]
        else:
            return DocumentCompressorPipeline(transformers=compressors)
    
    def _adjust_threshold_by_query(self, query: str) -> float:
        """
        Dynamically adjust similarity threshold based on query type.
        
        Args:
            query: The search query
            
        Returns:
            Adjusted threshold
        """
        query_lower = query.lower()
        
        # Lower threshold for specific value queries
        if any(term in query_lower for term in ["rate", "amount", "cost", "$", "percent"]):
            return max(0.3, self.config.similarity_threshold - 0.2)
        
        # Higher threshold for general policy queries
        if any(term in query_lower for term in ["policy", "procedure", "rule", "regulation"]):
            return min(0.8, self.config.similarity_threshold + 0.1)
        
        return self.config.similarity_threshold
    
    @BaseComponent.monitor_performance
    async def retrieve(
        self,
        query: str,
        k: Optional[int] = None
    ) -> List[Document]:
        """
        Retrieve and compress documents.
        
        Args:
            query: Search query
            k: Number of documents to return
            
        Returns:
            Compressed documents
        """
        try:
            # Adjust threshold based on query
            if hasattr(self.compressor, 'similarity_threshold'):
                original_threshold = self.compressor.similarity_threshold
                self.compressor.similarity_threshold = self._adjust_threshold_by_query(query)
            
            # Retrieve compressed documents
            if self.config.use_async:
                docs = await self.retriever.aget_relevant_documents(query)
            else:
                docs = self.retriever.get_relevant_documents(query)
            
            # Apply top_k limit if specified
            if k or self.config.top_k:
                limit = k or self.config.top_k
                docs = docs[:limit]
            
            # Log compression results
            self._log_event("compression_completed", {
                "query": query,
                "input_docs": len(docs),  # This is after compression
                "compression_mode": self.config.mode.value,
                "threshold": getattr(self.compressor, 'similarity_threshold', None)
            })
            
            return docs
            
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            # Fallback to base retriever
            return await self.base_retriever.aget_relevant_documents(query)
        
        finally:
            # Restore original threshold
            if hasattr(self.compressor, 'similarity_threshold') and 'original_threshold' in locals():
                self.compressor.similarity_threshold = original_threshold


def create_adaptive_compressor(
    base_retriever: BaseRetriever,
    llm: Optional[BaseLLM] = None,
    embeddings: Optional[Embeddings] = None,
    similarity_threshold: float = 0.5
) -> ContextualCompressionRetriever:
    """
    Create an adaptive compressor that chooses the best mode based on available resources.
    
    Args:
        base_retriever: Base retriever to compress
        llm: Optional language model
        embeddings: Optional embeddings model
        similarity_threshold: Similarity threshold for filtering
        
    Returns:
        Configured ContextualCompressionRetriever
    """
    # Determine best mode based on available resources
    if llm and embeddings:
        mode = CompressionMode.HYBRID
    elif llm:
        mode = CompressionMode.EXTRACTIVE
    elif embeddings:
        mode = CompressionMode.EMBEDDINGS_ONLY
    else:
        raise ValueError("Either LLM or embeddings required for compression")
    
    config = CompressionConfig(
        mode=mode,
        similarity_threshold=similarity_threshold
    )
    
    compressor = TravelContextualCompressor(
        base_retriever=base_retriever,
        llm=llm,
        embeddings=embeddings,
        config=config
    )
    
    return compressor.retriever


class QueryAwareCompressor(TravelContextualCompressor):
    """
    Extended compressor that adapts compression strategy based on query analysis.
    """
    
    def __init__(self, **kwargs):
        """Initialize with query analysis capabilities."""
        super().__init__(**kwargs)
        
        # Query type patterns
        self.query_patterns = {
            "specific_value": ["rate", "amount", "cost", "$", "how much", "price"],
            "policy": ["policy", "rule", "regulation", "allowed", "permitted"],
            "procedure": ["how to", "process", "steps", "procedure", "apply"],
            "comparison": ["difference", "versus", "vs", "compare", "better"],
            "eligibility": ["eligible", "qualify", "entitled", "can i", "am i"]
        }
    
    def _analyze_query_type(self, query: str) -> str:
        """Analyze query to determine type."""
        query_lower = query.lower()
        
        for query_type, patterns in self.query_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                return query_type
        
        return "general"
    
    def _get_mode_for_query_type(self, query_type: str) -> CompressionMode:
        """Get optimal compression mode for query type."""
        mode_mapping = {
            "specific_value": CompressionMode.EXTRACTIVE,
            "policy": CompressionMode.FILTER,
            "procedure": CompressionMode.HYBRID,
            "comparison": CompressionMode.EXTRACTIVE,
            "eligibility": CompressionMode.HYBRID,
            "general": CompressionMode.HYBRID
        }
        
        return mode_mapping.get(query_type, CompressionMode.HYBRID)
    
    async def retrieve(self, query: str, k: Optional[int] = None) -> List[Document]:
        """Retrieve with query-aware compression."""
        # Analyze query type
        query_type = self._analyze_query_type(query)
        
        # Temporarily adjust mode based on query
        original_mode = self.config.mode
        self.config.mode = self._get_mode_for_query_type(query_type)
        
        # Recreate compressor with new mode
        self.compressor = self._create_compressor()
        self.retriever = ContextualCompressionRetriever(
            base_compressor=self.compressor,
            base_retriever=self.base_retriever
        )
        
        try:
            # Log query analysis
            self._log_event("query_analyzed", {
                "query": query,
                "query_type": query_type,
                "compression_mode": self.config.mode.value
            })
            
            # Retrieve with adapted compression
            return await super().retrieve(query, k)
            
        finally:
            # Restore original mode
            self.config.mode = original_mode