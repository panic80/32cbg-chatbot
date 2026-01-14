"""Filtering strategies for the unified retrieval framework."""

from typing import List, Dict, Any, Optional, Set
import re
from datetime import datetime

from langchain_core.documents import Document

from app.unified_retrieval.strategies.base import BaseStrategy, RetrievalContext, StrategyType
from app.core.logging import get_logger

logger = get_logger(__name__)


class ContextAwareFilterStrategy(BaseStrategy):
    """
    Filter based on conversation context.
    
    This strategy filters documents based on the current conversation context,
    previous topics, and user preferences.
    """
    
    def __init__(
        self,
        context_window: int = 5,
        similarity_threshold: float = 0.7,
        **kwargs
    ):
        """
        Initialize the context-aware filter strategy.
        
        Args:
            context_window: Number of previous messages to consider
            similarity_threshold: Minimum similarity for context matching
        """
        super().__init__(
            strategy_type=StrategyType.FILTERING,
            name="context_aware_filter_strategy",
            required_inputs=["documents"],
            **kwargs
        )
        self.context_window = context_window
        self.similarity_threshold = similarity_threshold
    
    def _extract_context_keywords(self, context: RetrievalContext) -> Set[str]:
        """Extract relevant keywords from conversation context."""
        keywords = set()
        
        # Extract from conversation history if available
        if "conversation_history" in context.metadata:
            history = context.metadata["conversation_history"][-self.context_window:]
            for msg in history:
                # Extract nouns and important terms
                words = msg.get("content", "").lower().split()
                keywords.update(w for w in words if len(w) > 3)
        
        # Extract from current query
        query_words = context.get_query().lower().split()
        keywords.update(w for w in query_words if len(w) > 3)
        
        return keywords
    
    def _calculate_context_score(self, doc: Document, keywords: Set[str]) -> float:
        """Calculate how well a document matches the context."""
        doc_text = doc.page_content.lower()
        doc_words = set(doc_text.split())
        
        # Calculate overlap
        overlap = keywords.intersection(doc_words)
        if not keywords:
            return 1.0
        
        return len(overlap) / len(keywords)
    
    async def execute(self, context: RetrievalContext) -> RetrievalContext:
        """Execute the context-aware filtering."""
        documents = context.documents
        
        # Extract context keywords
        keywords = self._extract_context_keywords(context)
        
        # Filter documents based on context relevance
        filtered_docs = []
        for doc in documents:
            score = self._calculate_context_score(doc, keywords)
            if score >= self.similarity_threshold:
                filtered_docs.append(doc)
                # Store context score in metadata
                if "scores" not in doc.metadata:
                    doc.metadata["scores"] = {}
                doc.metadata["scores"]["context_score"] = score
        
        # Update context
        context.documents = filtered_docs
        
        # Add strategy output
        context.add_strategy_output(
            self.component_name,
            {
                "original_count": len(documents),
                "filtered_count": len(filtered_docs),
                "context_keywords": list(keywords)[:10],  # First 10 for logging
                "threshold": self.similarity_threshold
            }
        )
        
        self._log_event(
            "context_filtering_complete",
            {
                "docs_before": len(documents),
                "docs_after": len(filtered_docs),
                "keywords_used": len(keywords)
            }
        )
        
        return context


class RestrictionAwareFilterStrategy(BaseStrategy):
    """
    Prioritize restriction/limitation content.
    
    This strategy identifies and prioritizes documents that contain
    restrictions, limitations, or special conditions.
    """
    
    def __init__(
        self,
        restriction_keywords: Optional[List[str]] = None,
        boost_factor: float = 2.0,
        **kwargs
    ):
        """
        Initialize the restriction-aware filter strategy.
        
        Args:
            restriction_keywords: Keywords indicating restrictions
            boost_factor: How much to boost restriction documents
        """
        super().__init__(
            strategy_type=StrategyType.FILTERING,
            name="restriction_aware_filter_strategy",
            required_inputs=["documents"],
            **kwargs
        )
        self.restriction_keywords = restriction_keywords or self._get_default_keywords()
        self.boost_factor = boost_factor
    
    def _get_default_keywords(self) -> List[str]:
        """Get default restriction keywords."""
        return [
            "restriction", "limitation", "prohibited", "not authorized",
            "exception", "unless", "except", "only if", "must not",
            "cannot", "forbidden", "restricted", "limit", "maximum",
            "minimum", "condition", "requirement", "prerequisite",
            "subject to", "provided that", "contingent", "dependent"
        ]
    
    def _contains_restriction(self, doc: Document) -> bool:
        """Check if document contains restriction keywords."""
        content_lower = doc.page_content.lower()
        
        for keyword in self.restriction_keywords:
            if keyword.lower() in content_lower:
                return True
        
        # Check for patterns like "up to X km" or "maximum Y days"
        if re.search(r'\b(up to|maximum|no more than)\s+\d+\s*(km|kilometer|day|hour)', content_lower):
            return True
        
        return False
    
    async def execute(self, context: RetrievalContext) -> RetrievalContext:
        """Execute the restriction-aware filtering."""
        documents = context.documents
        
        # Separate restriction and non-restriction documents
        restriction_docs = []
        regular_docs = []
        
        for doc in documents:
            if self._contains_restriction(doc):
                # Mark as restriction document
                doc.metadata["contains_restriction"] = True
                doc.metadata["restriction_boost"] = self.boost_factor
                restriction_docs.append(doc)
            else:
                doc.metadata["contains_restriction"] = False
                regular_docs.append(doc)
        
        # Reorder: restrictions first
        context.documents = restriction_docs + regular_docs
        
        # Add strategy output
        context.add_strategy_output(
            self.component_name,
            {
                "total_docs": len(documents),
                "restriction_docs": len(restriction_docs),
                "regular_docs": len(regular_docs),
                "boost_factor": self.boost_factor
            }
        )
        
        self._log_event(
            "restriction_filtering_complete",
            {
                "restriction_docs_found": len(restriction_docs),
                "total_docs": len(documents)
            }
        )
        
        return context


class ClassAFilterStrategy(BaseStrategy):
    """
    Filter for Class A Reservist content.
    
    This strategy identifies and prioritizes content specifically
    relevant to Class A Reserve Force members.
    """
    
    def __init__(
        self,
        class_a_indicators: Optional[List[str]] = None,
        priority_boost: float = 1.5,
        **kwargs
    ):
        """
        Initialize the Class A filter strategy.
        
        Args:
            class_a_indicators: Terms indicating Class A content
            priority_boost: Boost factor for Class A content
        """
        super().__init__(
            strategy_type=StrategyType.FILTERING,
            name="class_a_filter_strategy",
            required_inputs=["documents"],
            **kwargs
        )
        self.class_a_indicators = class_a_indicators or self._get_default_indicators()
        self.priority_boost = priority_boost
    
    def _get_default_indicators(self) -> List[str]:
        """Get default Class A indicators."""
        return [
            "class a", "class-a", "reserve force", "reservist",
            "part-time", "part time", "res f", "reserve member",
            "class a reserve", "primary reserve", "supplementary reserve",
            "training period", "annual training", "weekend training",
            "class b", "class c"  # Include other classes for comparison
        ]
    
    def _get_class_relevance(self, doc: Document) -> Dict[str, Any]:
        """Determine document's relevance to Class A members."""
        content_lower = doc.page_content.lower()
        
        relevance = {
            "is_class_a": False,
            "mentions_class_a": False,
            "class_type": None,
            "confidence": 0.0
        }
        
        # Check for explicit Class A mentions
        if "class a" in content_lower or "class-a" in content_lower:
            relevance["is_class_a"] = True
            relevance["mentions_class_a"] = True
            relevance["class_type"] = "A"
            relevance["confidence"] = 1.0
        elif "class b" in content_lower or "class-b" in content_lower:
            relevance["class_type"] = "B"
            relevance["confidence"] = 0.8
        elif "class c" in content_lower or "class-c" in content_lower:
            relevance["class_type"] = "C"
            relevance["confidence"] = 0.8
        
        # Check for reserve force mentions
        for indicator in self.class_a_indicators:
            if indicator.lower() in content_lower:
                relevance["mentions_class_a"] = True
                if not relevance["class_type"]:
                    relevance["confidence"] = 0.6
                break
        
        return relevance
    
    async def execute(self, context: RetrievalContext) -> RetrievalContext:
        """Execute the Class A filtering."""
        documents = context.documents
        query_lower = context.get_query().lower()
        
        # Check if query mentions Class A
        query_mentions_class_a = any(
            indicator.lower() in query_lower 
            for indicator in self.class_a_indicators
        )
        
        # Process documents
        class_a_docs = []
        other_docs = []
        
        for doc in documents:
            relevance = self._get_class_relevance(doc)
            doc.metadata["class_relevance"] = relevance
            
            if query_mentions_class_a and relevance["is_class_a"]:
                # High priority for Class A docs when query asks for it
                doc.metadata["class_a_boost"] = self.priority_boost
                class_a_docs.append(doc)
            elif not query_mentions_class_a or relevance["mentions_class_a"]:
                # Include if query doesn't specify or doc mentions reserve
                other_docs.append(doc)
        
        # Reorder based on query intent
        if query_mentions_class_a:
            context.documents = class_a_docs + other_docs
        else:
            context.documents = documents  # Keep original order
        
        # Add strategy output
        context.add_strategy_output(
            self.component_name,
            {
                "total_docs": len(documents),
                "class_a_docs": len(class_a_docs),
                "query_mentions_class_a": query_mentions_class_a,
                "priority_boost": self.priority_boost
            }
        )
        
        return context


class MetadataFilterStrategy(BaseStrategy):
    """
    General metadata-based filtering.
    
    This strategy filters documents based on metadata fields
    and values specified in the context filters.
    """
    
    def __init__(
        self,
        strict_filtering: bool = False,
        allowed_fields: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize the metadata filter strategy.
        
        Args:
            strict_filtering: If True, all filters must match
            allowed_fields: List of metadata fields to consider
        """
        super().__init__(
            strategy_type=StrategyType.FILTERING,
            name="metadata_filter_strategy",
            required_inputs=["documents"],
            **kwargs
        )
        self.strict_filtering = strict_filtering
        self.allowed_fields = allowed_fields
    
    def _matches_filter(self, doc: Document, filters: Dict[str, Any]) -> bool:
        """Check if document matches the filters."""
        if not filters:
            return True
        
        matches = []
        
        for field, value in filters.items():
            # Skip if field not allowed
            if self.allowed_fields and field not in self.allowed_fields:
                continue
            
            # Check if field exists in metadata
            if field not in doc.metadata:
                matches.append(False)
                continue
            
            doc_value = doc.metadata[field]
            
            # Handle different value types
            if isinstance(value, list):
                # Any value in list
                matches.append(doc_value in value)
            elif isinstance(value, dict):
                # Handle complex filters (e.g., {"$gte": 5})
                if "$gte" in value:
                    matches.append(doc_value >= value["$gte"])
                elif "$lte" in value:
                    matches.append(doc_value <= value["$lte"])
                elif "$in" in value:
                    matches.append(doc_value in value["$in"])
                else:
                    matches.append(doc_value == value)
            else:
                # Exact match
                matches.append(str(doc_value).lower() == str(value).lower())
        
        # Apply AND or OR logic
        if self.strict_filtering:
            return all(matches) if matches else True
        else:
            return any(matches) if matches else True
    
    async def execute(self, context: RetrievalContext) -> RetrievalContext:
        """Execute the metadata filtering."""
        documents = context.documents
        filters = context.filters
        
        if not filters:
            # No filters to apply
            return context
        
        # Filter documents
        filtered_docs = [
            doc for doc in documents
            if self._matches_filter(doc, filters)
        ]
        
        # Update context
        context.documents = filtered_docs
        
        # Add strategy output
        context.add_strategy_output(
            self.component_name,
            {
                "original_count": len(documents),
                "filtered_count": len(filtered_docs),
                "filters_applied": filters,
                "strict_mode": self.strict_filtering
            }
        )
        
        self._log_event(
            "metadata_filtering_complete",
            {
                "docs_before": len(documents),
                "docs_after": len(filtered_docs),
                "filters": list(filters.keys())
            }
        )
        
        return context