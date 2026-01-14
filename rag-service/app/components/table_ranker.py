"""Table-specific document ranker for improved retrieval accuracy."""

import re
from typing import List, Dict, Any, Tuple, Optional
from langchain_core.documents import Document

from app.core.logging import get_logger
from app.utils.table_validator import TableValidator

logger = get_logger(__name__)


class TableRanker:
    """Ranks documents with special consideration for table content."""
    
    def __init__(self):
        """Initialize the table ranker."""
        self.table_content_types = {
            "table_markdown", "table_key_value", "table_json", 
            "table_summary", "table_unstructured", "table_html"
        }
        
    def rank_documents(
        self, 
        documents: List[Document], 
        query: str,
        query_type: Optional[str] = None,
        value_patterns: Optional[List[str]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Rank documents with table-specific scoring.
        
        Args:
            documents: List of documents to rank
            query: Original query
            query_type: Type of query (e.g., "table", "simple")
            value_patterns: Extracted numeric patterns from query
            
        Returns:
            List of (document, score) tuples sorted by relevance
        """
        scored_docs = []
        
        for doc in documents:
            score = self._calculate_score(doc, query, query_type, value_patterns)
            scored_docs.append((doc, score))
        
        # Sort by score descending
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return scored_docs
    
    def _calculate_score(
        self,
        doc: Document,
        query: str,
        query_type: Optional[str] = None,
        value_patterns: Optional[List[str]] = None
    ) -> float:
        """Calculate relevance score for a document."""
        score = 1.0  # Base score
        
        metadata = doc.metadata
        content = doc.page_content.lower()
        query_lower = query.lower()
        
        # 1. Boost table documents for table queries
        if query_type == "table" or self._is_table_query(query):
            if metadata.get("content_type") in self.table_content_types:
                score *= 2.0
                logger.debug(f"Table content boost applied: {metadata.get('content_type')}")
        
        # 2. Exact value matching and dollar amounts
        if value_patterns:
            for pattern in value_patterns:
                if pattern.lower() in content:
                    score *= 3.0  # Strong boost for exact value match
                    logger.debug(f"Exact value match found: {pattern}")
                    
                    # Extra boost if it's in a key-value pair
                    if metadata.get("content_type") == "table_key_value":
                        if pattern == metadata.get("value", "").lower():
                            score *= 2.0
                            logger.debug(f"Key-value exact match: {pattern}")
        
        # Special handling for dollar amounts in meal queries
        if self._is_meal_query(query) and "$" in content:
            # Count dollar signs
            dollar_count = content.count("$")
            if dollar_count > 0:
                score *= (1.5 + (0.1 * min(dollar_count, 10)))  # Boost based on number of dollar amounts
                logger.debug(f"Dollar amounts found: {dollar_count}")
                
                # Extra boost if table contains meal-related terms AND dollar amounts
                meal_terms = ["breakfast", "lunch", "dinner", "meal"]
                if any(term in content for term in meal_terms):
                    score *= 1.5
                    logger.debug("Meal table with dollar amounts detected")
        
        # 3. Header matching
        headers = metadata.get("headers", [])
        if headers:
            query_words = set(query_lower.split())
            header_words = set(" ".join(headers).lower().split())
            
            # Check for header word matches
            header_matches = query_words.intersection(header_words)
            if header_matches:
                score *= 1.5 * len(header_matches)
                logger.debug(f"Header matches: {header_matches}")
        
        # 4. Location matching
        locations = self._extract_locations(query)
        if locations:
            for location in locations:
                if location.lower() in content:
                    score *= 1.5
                    logger.debug(f"Location match: {location}")
                    
                    # Extra boost if in title
                    title = metadata.get("table_title", "").lower()
                    if location.lower() in title:
                        score *= 1.5
                        logger.debug(f"Location in title: {location}")
        
        # 5. Table title relevance
        table_title = metadata.get("table_title", "").lower()
        if table_title:
            # Calculate word overlap
            title_words = set(table_title.split())
            query_words = set(query_lower.split())
            overlap = title_words.intersection(query_words)
            
            if overlap:
                score *= 1.2 * len(overlap)
                logger.debug(f"Title word overlap: {overlap}")
        
        # 6. Numeric value extraction and validation
        if self._contains_numeric_query(query):
            # Check if document contains numeric values
            numeric_values = self._extract_numeric_values(content)
            if numeric_values:
                score *= 1.3
                logger.debug(f"Document contains {len(numeric_values)} numeric values")
        
        # 7. Query term density
        query_terms = query_lower.split()
        term_count = sum(1 for term in query_terms if term in content)
        if term_count > 0:
            density_score = min(term_count / len(query_terms), 1.0)
            score *= (1 + density_score)
            logger.debug(f"Term density score: {density_score}")
        
        # 8. Continuation penalty (for chunked tables)
        if metadata.get("is_continuation", False):
            score *= 0.8  # Slight penalty for continuation chunks
            logger.debug("Continuation chunk penalty applied")
        
        # 9. Content type specific scoring
        content_type = metadata.get("content_type", "")
        if content_type == "table_key_value":
            # Best for exact value lookups
            if value_patterns:
                score *= 1.2
        elif content_type == "table_json":
            # Good for structured queries
            score *= 1.1
        elif content_type == "table_summary":
            # Good for general table understanding
            score *= 1.0
        
        return score
    
    def _is_table_query(self, query: str) -> bool:
        """Determine if query is likely seeking table data."""
        query_lower = query.lower()
        
        table_indicators = [
            "rate", "rates", "limit", "limits", "allowance", "allowances",
            "maximum", "minimum", "per", "table", "schedule", "appendix",
            "how much", "what is the", "cost", "price", "amount"
        ]
        
        return any(indicator in query_lower for indicator in table_indicators)
    
    def _is_meal_query(self, query: str) -> bool:
        """Determine if query is about meal allowances."""
        query_lower = query.lower()
        
        meal_indicators = [
            "meal", "breakfast", "lunch", "dinner", "food",
            "per meal", "meal allowance", "meal rate"
        ]
        
        return any(indicator in query_lower for indicator in meal_indicators)
    
    def _extract_locations(self, query: str) -> List[str]:
        """Extract location names from query."""
        locations = []
        query_lower = query.lower()
        
        # Canadian provinces and territories
        provinces = [
            "ontario", "quebec", "british columbia", "alberta", "manitoba",
            "saskatchewan", "nova scotia", "new brunswick", "newfoundland",
            "prince edward island", "northwest territories", "yukon", "nunavut"
        ]
        
        # Check for provinces
        for province in provinces:
            if province in query_lower:
                locations.append(province)
        
        return locations
    
    def _contains_numeric_query(self, query: str) -> bool:
        """Check if query contains numeric patterns."""
        numeric_patterns = [
            r'\d+',  # Any number
            r'\$[\d,]+',  # Dollar amounts
            r'\d+\s*(?:km|miles?|cents?|dollars?)',  # Units
            r'\d+\.?\d*\s*%',  # Percentages
        ]
        
        for pattern in numeric_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        
        return False
    
    def _extract_numeric_values(self, text: str) -> List[float]:
        """Extract numeric values from text."""
        values = []
        
        # Split text into words
        words = text.split()
        
        for word in words:
            # Try to extract numeric value
            numeric_val = TableValidator.extract_numeric_value(word)
            if numeric_val is not None:
                values.append(numeric_val)
        
        return values
    
    def filter_and_rerank(
        self,
        documents: List[Document],
        query: str,
        top_k: int = 10,
        query_type: Optional[str] = None,
        value_patterns: Optional[List[str]] = None
    ) -> List[Document]:
        """
        Filter and rerank documents for optimal retrieval.
        
        Args:
            documents: Input documents
            query: Search query
            top_k: Number of documents to return
            query_type: Type of query
            value_patterns: Extracted value patterns
            
        Returns:
            Reranked documents
        """
        # Rank all documents
        ranked_docs = self.rank_documents(documents, query, query_type, value_patterns)
        
        # Log top scores for debugging
        if ranked_docs:
            logger.info(f"Top 3 document scores: {[(doc.metadata.get('content_type', 'unknown'), score) for doc, score in ranked_docs[:3]]}")
        
        # Return top k documents
        return [doc for doc, score in ranked_docs[:top_k]]