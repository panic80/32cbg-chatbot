"""Restriction-aware retriever for policy limitations and restrictions."""
import logging
import re
from typing import List, Optional, ClassVar
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

logger = logging.getLogger(__name__)


class RestrictionAwareRetriever(BaseRetriever):
    """
    Retriever that prioritizes documents containing restrictions, limitations, and distance constraints.
    """
    
    base_retriever: BaseRetriever
    boost_factor: float = 2.0
    
    # Keywords indicating restrictions
    restriction_keywords: ClassVar[List[str]] = [
        "restriction", "limit", "limitation", "maximum", "exceed",
        "not authorized", "prohibited", "must not", "cannot",
        "up to", "no more than", "within"
    ]
    
    # Distance/measurement patterns
    distance_patterns: ClassVar[List[str]] = [
        r"\d+\s*km", r"\d+\s*kilometer", r"\d+\s*kilometre",
        r"\d+\s*mile", r"\d+\s*nautical"
    ]
    
    # Class designations
    class_patterns: ClassVar[List[str]] = [
        r"Class\s+[ABC]", r"class\s+[abc]", 
        r"Reserve\s+Force", r"Primary\s+Reserve"
    ]
    
    def _calculate_restriction_score(self, doc: Document) -> float:
        """Calculate a score based on restriction-related content."""
        content_lower = doc.page_content.lower()
        score = 1.0
        
        # Check for restriction keywords
        restriction_count = sum(1 for keyword in self.restriction_keywords 
                              if keyword.lower() in content_lower)
        if restriction_count > 0:
            score *= (1 + 0.3 * restriction_count)  # 30% boost per keyword
        
        # Check for distance patterns
        for pattern in self.distance_patterns:
            if re.search(pattern, doc.page_content, re.IGNORECASE):
                score *= 1.5  # 50% boost for distance mentions
                doc.metadata["has_distance_restriction"] = True
        
        # Check for class designations near restrictions
        has_class = any(re.search(pattern, doc.page_content, re.IGNORECASE) 
                       for pattern in self.class_patterns)
        has_restriction = restriction_count > 0
        
        if has_class and has_restriction:
            score *= 1.5  # Additional 50% boost for class-specific restrictions
            doc.metadata["has_class_restriction"] = True
        
        # Extra boost for PMV/vehicle restrictions
        if any(term in content_lower for term in ["pmv", "motor vehicle", "drive", "driving"]):
            if has_restriction or "has_distance_restriction" in doc.metadata:
                score *= 1.3  # 30% boost for vehicle-related restrictions
        
        return score
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        """Get documents with restriction scoring."""
        # Enhance query to look for restrictions
        enhanced_query = query
        query_lower = query.lower()
        
        # Add restriction context if not present
        if not any(keyword in query_lower for keyword in ["restriction", "limit", "maximum"]):
            if any(term in query_lower for term in ["drive", "pmv", "vehicle", "car"]):
                enhanced_query = f"{query} restrictions limitations authorization"
        
        # Get base documents
        docs = self.base_retriever.get_relevant_documents(
            enhanced_query,
            callbacks=run_manager.get_child() if run_manager else None
        )
        
        # Score and sort documents
        scored_docs = []
        for doc in docs:
            score = self._calculate_restriction_score(doc)
            doc.metadata["restriction_score"] = score
            scored_docs.append((score, doc))
        
        # Sort by restriction score
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        # Log results
        high_score_count = sum(1 for score, _ in scored_docs if score > 1.5)
        logger.info(f"Restriction Retriever: Found {high_score_count}/{len(docs)} high-restriction documents")
        
        return [doc for _, doc in scored_docs]
    
    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        """Async version."""
        return self._get_relevant_documents(query, run_manager=run_manager)
