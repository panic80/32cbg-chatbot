"""Document store service for managing documents."""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
import asyncio

from app.core.vectorstore import VectorStoreManager
from app.core.config import settings
from app.core.logging import get_logger
from app.models.documents import (
    Document, DocumentSearchRequest, DocumentSearchResult,
    DocumentListResponse
)
from app.models.query import Source
from app.services.cache import CacheService, QueryCache
from app.services.source_repository import SourceRepository

logger = get_logger(__name__)


class DocumentStore:
    """Service for document storage and retrieval."""
    
    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        cache_service: Optional[CacheService] = None,
        source_repository: Optional[SourceRepository] = None
    ):
        """Initialize document store."""
        self.vector_store = vector_store_manager
        self.cache_service = cache_service
        self.query_cache = QueryCache(cache_service) if cache_service else None
        self.source_repository = source_repository
        
    async def search(
        self,
        request: DocumentSearchRequest
    ) -> List[DocumentSearchResult]:
        """Search for documents."""
        try:
            # Check cache first
            if self.query_cache:
                cached = await self.query_cache.get_results(
                    request.query,
                    request.filters
                )
                if cached:
                    logger.info("Query cache hit")
                    return self._deserialize_results(cached)
                    
            # Perform vector search
            results = await self.vector_store.search(
                query=request.query,
                k=request.limit,
                filter_dict=request.filters,
                search_type=settings.retrieval_search_type
            )
            
            # Convert to response format
            search_results = []
            for doc, score in results:
                # Extract document from metadata
                doc_result = DocumentSearchResult(
                    document=Document(
                        id=doc.metadata.get("id", ""),
                        content=doc.page_content,
                        metadata=doc.metadata,
                        created_at=doc.metadata.get("created_at", datetime.now(timezone.utc))
                    ),
                    score=score if request.include_scores else None,
                    highlights=self._extract_highlights(doc.page_content, request.query)
                )
                search_results.append(doc_result)
                
            # Cache results
            if self.query_cache:
                await self.query_cache.set_results(
                    request.query,
                    self._serialize_results(search_results),
                    request.filters
                )
                
            return search_results
            
        except Exception as e:
            logger.error(f"Document search failed: {e}")
            raise
            
    async def get_by_id(self, document_id: str) -> Optional[Document]:
        """Get document by ID."""
        try:
            # Search by metadata filter
            results = await self.vector_store.search(
                query="",  # Empty query
                k=1,
                filter_dict={"id": document_id}
            )
            
            if results:
                doc, _ = results[0]
                return Document(
                    id=doc.metadata.get("id", ""),
                    content=doc.page_content,
                    metadata=doc.metadata,
                    created_at=doc.metadata.get("created_at", datetime.now(timezone.utc))
                )
                
            return None
            
        except Exception as e:
            logger.error(f"Failed to get document {document_id}: {e}")
            return None
            
    async def list_documents(
        self,
        page: int = 1,
        page_size: int = 20,
        filters: Optional[Dict[str, Any]] = None
    ) -> DocumentListResponse:
        """List documents with pagination."""
        try:
            # Get collection stats
            stats = await self.vector_store.get_collection_stats()
            total_count = stats.get("document_count", 0)
            
            # For now, return empty list as we need to implement proper listing
            # This would require additional metadata storage
            return DocumentListResponse(
                documents=[],
                total=total_count,
                page=page,
                page_size=page_size
            )
            
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            raise
            
    async def delete_by_id(self, document_id: str) -> bool:
        """Delete document by ID."""
        try:
            # Find all chunks with this parent ID
            results = await self.vector_store.search(
                query="",
                k=1000,  # Get all chunks
                filter_dict={"parent_id": document_id}
            )
            
            # Extract chunk IDs
            chunk_ids = [doc.metadata.get("id") for doc, _ in results]
            
            if chunk_ids:
                # Delete from vector store
                success = await self.vector_store.delete_documents(ids=chunk_ids)
                
                # Clear from cache
                if self.cache_service:
                    await self.cache_service.delete(f"doc:{document_id}")
                    
                return success
                
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False
            
    def convert_to_sources(
        self,
        documents: List[Tuple[Any, float]],
        max_sources: int = 5
    ) -> List[Source]:
        """Convert search results to sources for citations."""
        sources = []
        
        for doc, score in documents[:max_sources]:
            metadata = doc.metadata.model_dump() if hasattr(doc.metadata, 'model_dump') else doc.metadata
            canonical_url = metadata.get("canonical_url") or metadata.get("source")
            text_preview = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content

            source = Source(
                id=metadata.get("id", ""),
                source_id=metadata.get("source_id"),
                text=text_preview,
                title=metadata.get("title"),
                url=canonical_url,
                section=metadata.get("section"),
                page=metadata.get("page_number"),
                score=score,
                metadata={
                    "type": metadata.get("type"),
                    "last_modified": metadata.get("last_modified"),
                    "tags": metadata.get("tags", [])
                }
            )
            sources.append(source)
        
        return sources
        
    def _extract_highlights(self, content: str, query: str) -> List[str]:
        """Extract highlighted snippets from content."""
        highlights = []
        
        # Simple keyword highlighting
        query_terms = query.lower().split()
        sentences = content.split(". ")
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(term in sentence_lower for term in query_terms):
                # Trim to reasonable length
                if len(sentence) > 200:
                    sentence = sentence[:200] + "..."
                highlights.append(sentence)
                
                if len(highlights) >= 3:
                    break
                    
        return highlights
        
    def _serialize_results(self, results: List[DocumentSearchResult]) -> Dict:
        """Serialize results for caching."""
        return {
            "results": [
                {
                    "document": {
                        "id": r.document.id,
                        "content": r.document.content,
                        "metadata": r.document.metadata.model_dump() if hasattr(r.document.metadata, 'model_dump') else r.document.metadata,
                        "created_at": r.document.created_at.isoformat()
                    },
                    "score": r.score,
                    "highlights": r.highlights
                }
                for r in results
            ],
            "cached_at": datetime.now(timezone.utc).isoformat()
        }
        
    def _deserialize_results(self, cached: Dict) -> List[DocumentSearchResult]:
        """Deserialize cached results."""
        results = []
        
        for item in cached.get("results", []):
            doc_data = item["document"]
            result = DocumentSearchResult(
                document=Document(
                    id=doc_data["id"],
                    content=doc_data["content"],
                    metadata=doc_data["metadata"],
                    created_at=datetime.fromisoformat(doc_data["created_at"])
                ),
                score=item.get("score"),
                highlights=item.get("highlights", [])
            )
            results.append(result)
            
        return results
