"""Class A Reservist-specific retriever component."""
import logging
from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

logger = logging.getLogger(__name__)


class ClassARetriever(BaseRetriever):
    """
    Retriever that boosts and filters content relevant to Class A Reservists.
    """
    
    base_retriever: BaseRetriever
    boost_keywords: List[str] = [
        "Class A", "class a", "CLASS A",
        "Primary Reserve", "primary reserve",
        "part-time", "part time",
        "weekend training", "parade night",
        "12 days", "35 days",
        "Class A/B", "A/B service",
        "reservist", "Reservist"
    ]
    boost_factor: float = 1.5
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        """Get documents with Class A boosting."""
        # Add Class A context to query if not already present
        enhanced_query = query
        if not any(keyword.lower() in query.lower() for keyword in ["class a", "reservist"]):
            enhanced_query = f"{query} (Class A Reserve context)"
        
        # Get base documents
        docs = self.base_retriever.get_relevant_documents(
            enhanced_query,
            callbacks=run_manager.get_child() if run_manager else None
        )
        
        # Score and boost documents
        scored_docs = []
        for doc in docs:
            score = 1.0
            content_lower = doc.page_content.lower()
            
            # Boost documents that mention Class A concepts
            for keyword in self.boost_keywords:
                if keyword.lower() in content_lower:
                    score *= self.boost_factor
                    # Add metadata to indicate Class A relevance
                    if "class_a_relevant" not in doc.metadata:
                        doc.metadata["class_a_relevant"] = True
                    break
            
            # Extra boost for explicit Class A sections
            if any(phrase in content_lower for phrase in [
                "for class a", "class a members", "class a reservists",
                "class a primary reserve", "class a service"
            ]):
                score *= 1.2
            
            doc.metadata["relevance_score"] = score
            scored_docs.append((score, doc))
        
        # Sort by score and return
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        # Log boosting results
        boosted_count = sum(1 for score, _ in scored_docs if score > 1.0)
        logger.info(f"Class A Retriever: Boosted {boosted_count}/{len(docs)} documents")
        
        return [doc for _, doc in scored_docs]
    
    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        """Async version."""
        # For now, just call sync version
        return self._get_relevant_documents(query, run_manager=run_manager)
