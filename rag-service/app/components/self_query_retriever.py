"""
Self-Query Retriever for structured queries with metadata filtering.

This module implements LangChain's SelfQueryRetriever to enable natural language
queries that are automatically parsed into filters for structured retrieval.
"""

from typing import List, Optional, Dict, Any, Sequence
import logging
from datetime import datetime
from dataclasses import dataclass

from langchain_core.language_models import BaseLLM
from langchain_core.documents import Document
from langchain_classic.chains.query_constructor.base import AttributeInfo
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever as LangChainSelfQueryRetriever
from langchain_core.vectorstores import VectorStore
from pydantic import Field, BaseModel

from app.components.base import BaseComponent
from app.utils.retry import with_retry_async

logger = logging.getLogger(__name__)


# Define metadata schema for travel documents
TRAVEL_DOCUMENT_ATTRIBUTES = [
    AttributeInfo(
        name="source",
        description="Source of the document (e.g., NJC, department policy)",
        type="string"
    ),
    AttributeInfo(
        name="document_type",
        description="Type of document (e.g., policy, directive, guide, form)",
        type="string"
    ),
    AttributeInfo(
        name="section",
        description="Section or chapter of the document",
        type="string"
    ),
    AttributeInfo(
        name="effective_date",
        description="Date when the document became effective",
        type="string"  # Store as ISO format string
    ),
    AttributeInfo(
        name="last_updated",
        description="Date when the document was last updated",
        type="string"
    ),
    AttributeInfo(
        name="jurisdiction",
        description="Jurisdiction (e.g., domestic, international, specific province)",
        type="string"
    ),
    AttributeInfo(
        name="travel_type",
        description="Type of travel (e.g., duty_travel, relocation, training)",
        type="string"
    ),
    AttributeInfo(
        name="has_rates",
        description="Whether the document contains specific rates or amounts",
        type="boolean"
    ),
    AttributeInfo(
        name="topic",
        description="Main topic (e.g., meals, accommodation, transportation)",
        type="string"
    )
]

# Document content description
TRAVEL_CONTENT_DESCRIPTION = """
Canadian Forces travel instruction documents containing policies, procedures,
rates, and allowances for military personnel travel. Documents cover topics
including meal allowances, accommodation, transportation, and incidental expenses.
"""


class SelfQueryConfig(BaseModel):
    """Configuration for self-query retriever."""
    enable_limit: bool = Field(default=True, description="Allow queries to specify result limit")
    max_results: int = Field(default=20, description="Maximum results to return")
    default_results: int = Field(default=5, description="Default number of results")
    verbose: bool = Field(default=False, description="Enable verbose logging")
    use_compression: bool = Field(default=True, description="Compress results after filtering")


class TravelSelfQueryRetriever(BaseComponent):
    """
    Self-query retriever optimized for travel instruction queries.
    
    Features:
    - Natural language to filter conversion
    - Date range queries
    - Topic-based filtering
    - Rate/amount filtering
    - Jurisdiction-aware search
    """
    
    def __init__(
        self,
        vectorstore: VectorStore,
        llm: BaseLLM,
        config: Optional[SelfQueryConfig] = None,
        metadata_field_info: Optional[Sequence[AttributeInfo]] = None,
        document_contents: Optional[str] = None
    ):
        """
        Initialize the self-query retriever.
        
        Args:
            vectorstore: Vector store with metadata filtering support
            llm: Language model for query parsing
            config: Configuration options
            metadata_field_info: Custom metadata schema
            document_contents: Custom document description
        """
        super().__init__(component_type="self_query_retriever")
        
        self.vectorstore = vectorstore
        self.llm = llm
        self.config = config or SelfQueryConfig()
        
        # Use provided metadata info or defaults
        metadata_info = metadata_field_info or TRAVEL_DOCUMENT_ATTRIBUTES
        doc_contents = document_contents or TRAVEL_CONTENT_DESCRIPTION
        
        # Create the LangChain self-query retriever
        self.retriever = LangChainSelfQueryRetriever.from_llm(
            llm=llm,
            vectorstore=vectorstore,
            document_contents=doc_contents,
            metadata_field_info=metadata_info,
            enable_limit=self.config.enable_limit,
            verbose=self.config.verbose
        )
        
        # Set search kwargs
        self.retriever.search_kwargs = {
            "k": self.config.default_results
        }
    
    def _preprocess_query(self, query: str) -> str:
        """
        Preprocess query to improve parsing.
        
        Args:
            query: Original query
            
        Returns:
            Preprocessed query
        """
        # Add context for better parsing
        context_additions = []
        
        query_lower = query.lower()
        
        # Add temporal context
        if "latest" in query_lower or "current" in query_lower:
            context_additions.append("most recently updated")
        
        # Add jurisdiction hints
        if any(term in query_lower for term in ["domestic", "canada", "within"]):
            context_additions.append("for domestic travel")
        elif any(term in query_lower for term in ["international", "foreign", "abroad"]):
            context_additions.append("for international travel")
        
        # Add document type hints
        if any(term in query_lower for term in ["form", "template", "document"]):
            context_additions.append("document type is form")
        elif any(term in query_lower for term in ["policy", "directive", "regulation"]):
            context_additions.append("document type is policy")
        
        # Combine with original query
        if context_additions:
            return f"{query} ({', '.join(context_additions)})"
        
        return query
    
    @BaseComponent.monitor_performance
    @with_retry_async()
    async def retrieve(
        self,
        query: str,
        k: Optional[int] = None
    ) -> List[Document]:
        """
        Retrieve documents using self-query.
        
        Args:
            query: Natural language query with optional filters
            k: Number of results to return
            
        Returns:
            Filtered and relevant documents
        """
        try:
            # Preprocess query
            processed_query = self._preprocess_query(query)
            
            # Update k if provided
            if k:
                self.retriever.search_kwargs["k"] = min(k, self.config.max_results)
            
            # Log query parsing attempt
            self.log_event("self_query_start", {
                "original_query": query,
                "processed_query": processed_query,
                "k": self.retriever.search_kwargs.get("k")
            })
            
            # Retrieve with structured query
            docs = await self.retriever.aget_relevant_documents(processed_query)
            
            # Log successful retrieval
            self.log_event("self_query_success", {
                "query": query,
                "num_results": len(docs),
                "has_filters": processed_query != query
            })
            
            return docs
            
        except Exception as e:
            logger.error(f"Self-query failed: {e}")
            
            # Fallback to simple vector search
            self.log_event("self_query_fallback", {
                "query": query,
                "error": str(e)
            })
            
            # Use direct vector search as fallback
            k_value = k or self.config.default_results
            results = await self.vectorstore.asimilarity_search(query, k=k_value)
            
            return results
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Synchronous version of retrieve."""
        # Update search kwargs
        self.retriever.search_kwargs["k"] = self.config.default_results
        
        try:
            processed_query = self._preprocess_query(query)
            return self.retriever.get_relevant_documents(processed_query)
        except Exception as e:
            logger.error(f"Self-query failed: {e}")
            # Fallback to simple search
            return self.vectorstore.similarity_search(
                query, 
                k=self.config.default_results
            )


class EnhancedSelfQueryRetriever(TravelSelfQueryRetriever):
    """
    Enhanced self-query retriever with additional features.
    
    Adds:
    - Query examples for better parsing
    - Custom filter validation
    - Result post-processing
    """
    
    # Example queries for different filter types
    EXAMPLE_QUERIES = {
        "date_range": "Show me meal rates updated after January 2024",
        "jurisdiction": "Find accommodation policies for international travel",
        "document_type": "Get all forms related to travel claims",
        "topic": "What are the transportation allowances?",
        "combined": "Find recent policies about meal allowances for domestic travel"
    }
    
    def __init__(self, **kwargs):
        """Initialize with enhanced features."""
        super().__init__(**kwargs)
        
        # Add example queries to prompt
        self._enhance_retriever_prompt()
    
    def _enhance_retriever_prompt(self):
        """Add examples to improve query parsing."""
        # This would modify the retriever's prompt to include examples
        # Implementation depends on LangChain version
        pass
    
    def _validate_filters(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize extracted filters.
        
        Args:
            filters: Extracted filters from query
            
        Returns:
            Validated filters
        """
        validated = {}
        
        for key, value in filters.items():
            # Validate date fields
            if key in ["effective_date", "last_updated"] and isinstance(value, str):
                try:
                    # Ensure ISO format
                    datetime.fromisoformat(value.replace('Z', '+00:00'))
                    validated[key] = value
                except ValueError:
                    logger.warning(f"Invalid date format for {key}: {value}")
                    continue
            
            # Validate boolean fields
            elif key == "has_rates":
                validated[key] = bool(value)
            
            # Validate string enums
            elif key in ["document_type", "jurisdiction", "travel_type", "topic"]:
                # Normalize to lowercase
                validated[key] = str(value).lower()
            
            else:
                validated[key] = value
        
        return validated
    
    def _post_process_results(
        self,
        docs: List[Document],
        query: str
    ) -> List[Document]:
        """
        Post-process results to ensure quality.
        
        Args:
            docs: Retrieved documents
            query: Original query
            
        Returns:
            Processed documents
        """
        # Sort by relevance indicators
        query_lower = query.lower()
        
        # Score based on metadata match quality
        for doc in docs:
            score = 0.0
            metadata = doc.metadata
            
            # Boost recent documents for "latest" queries
            if "latest" in query_lower or "current" in query_lower:
                if "last_updated" in metadata:
                    try:
                        update_date = datetime.fromisoformat(
                            metadata["last_updated"].replace('Z', '+00:00')
                        )
                        days_old = (datetime.now() - update_date).days
                        if days_old < 30:
                            score += 2.0
                        elif days_old < 90:
                            score += 1.0
                    except:
                        pass
            
            # Boost exact topic matches
            if "topic" in metadata and metadata["topic"] in query_lower:
                score += 1.5
            
            # Store score
            doc.metadata["self_query_score"] = score
        
        # Sort by score
        docs.sort(
            key=lambda d: d.metadata.get("self_query_score", 0),
            reverse=True
        )
        
        return docs
    
    async def retrieve(
        self,
        query: str,
        k: Optional[int] = None
    ) -> List[Document]:
        """Retrieve with enhanced processing."""
        # Get base results
        docs = await super().retrieve(query, k)
        
        # Post-process results
        docs = self._post_process_results(docs, query)
        
        return docs


def create_travel_self_query_retriever(
    vectorstore: VectorStore,
    llm: BaseLLM,
    enhanced: bool = True,
    **kwargs
) -> TravelSelfQueryRetriever:
    """
    Factory function to create a travel-optimized self-query retriever.
    
    Args:
        vectorstore: Vector store with metadata support
        llm: Language model for parsing
        enhanced: Whether to use enhanced version
        **kwargs: Additional configuration
        
    Returns:
        Configured self-query retriever
    """
    config = SelfQueryConfig(**kwargs) if kwargs else SelfQueryConfig()
    
    if enhanced:
        return EnhancedSelfQueryRetriever(
            vectorstore=vectorstore,
            llm=llm,
            config=config,
            metadata_field_info=TRAVEL_DOCUMENT_ATTRIBUTES,
            document_contents=TRAVEL_CONTENT_DESCRIPTION
        )
    else:
        return TravelSelfQueryRetriever(
            vectorstore=vectorstore,
            llm=llm,
            config=config,
            metadata_field_info=TRAVEL_DOCUMENT_ATTRIBUTES,
            document_contents=TRAVEL_CONTENT_DESCRIPTION
        )