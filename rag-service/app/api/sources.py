"""Source management API endpoints."""

from datetime import datetime
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from app.core.logging import get_logger
from app.models.documents import (
    DocumentSearchRequest,
    DocumentSearchResult,
)
from app.api.security import verify_admin_bearer_token

logger = get_logger(__name__)

router = APIRouter(dependencies=[Depends(verify_admin_bearer_token)])


class SourceSummaryResponse(BaseModel):
    """Summarises a canonical source entry."""

    source_id: str
    title: Optional[str] = None
    canonical_url: Optional[str] = None
    reference_path: Optional[str] = None
    document_type: Optional[str] = None
    section: Optional[str] = None
    chunk_count: int = 0
    document_count: int = 0
    last_ingested_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SourceListResponse(BaseModel):
    """Paginated list of canonical sources."""

    items: List[SourceSummaryResponse]
    total: int
    page: int
    page_size: int


class SourceDetailResponse(SourceSummaryResponse):
    """Detailed view of a canonical source, including linked chunks."""

    chunks: List[Dict[str, Any]] = Field(default_factory=list)


class QuerySourceResponse(BaseModel):
    """Represents a source cited in a specific query response."""

    source_id: str
    rank: Optional[int] = None
    snippet: Optional[str] = None
    title: Optional[str] = None
    canonical_url: Optional[str] = None
    reference_path: Optional[str] = None


@router.post("/sources/search")
async def search_sources(
    request: Request,
    search_request: DocumentSearchRequest
) -> List[DocumentSearchResult]:
    """Search for document sources via vector semantics."""
    try:
        document_store = request.app.state.document_store
        return await document_store.search(search_request)
    except Exception as exc:
        logger.error("Source search failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/sources", response_model=SourceListResponse)
async def list_sources(
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    document_type: Optional[str] = None,
    search: Optional[str] = None
) -> SourceListResponse:
    """List canonical sources with pagination."""
    source_repo = getattr(request.app.state, "source_repository", None)
    if not source_repo:
        raise HTTPException(status_code=503, detail="Source repository unavailable")

    try:
        filters: Dict[str, Any] = {}
        if document_type:
            filters["document_type"] = document_type
        if search:
            filters["search"] = search

        offset = (page - 1) * page_size
        records, total = await source_repo.list_sources(
            offset=offset,
            limit=page_size,
            filters=filters
        )

        items = [SourceSummaryResponse(**record) for record in records]
        return SourceListResponse(
            items=items,
            total=total,
            page=page,
            page_size=page_size
        )
    except Exception as exc:
        logger.error("Source listing failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/sources/stats")
async def get_source_stats(request: Request) -> Dict[str, Any]:
    """Return aggregate statistics about canonical sources."""
    source_repo = getattr(request.app.state, "source_repository", None)
    if not source_repo:
        raise HTTPException(status_code=503, detail="Source repository unavailable")

    try:
        stats = await source_repo.get_statistics()
        return stats
    except Exception as exc:
        logger.error("Stats retrieval failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/sources/count")
async def get_source_count(request: Request) -> Dict[str, Any]:
    """Return total number of canonical sources."""
    source_repo = getattr(request.app.state, "source_repository", None)
    if not source_repo:
        raise HTTPException(status_code=503, detail="Source repository unavailable")

    try:
        stats = await source_repo.get_statistics()
        return {"count": stats.get("total_sources", 0), "status": "success"}
    except Exception as exc:
        logger.error("Count retrieval failed: %s", exc)
        return {"count": 0, "status": "error", "message": str(exc)}


@router.get("/sources/{source_id}", response_model=SourceDetailResponse)
async def get_source(
    request: Request,
    source_id: str
) -> SourceDetailResponse:
    """Fetch canonical source details, falling back to raw document lookup if needed."""
    source_repo = getattr(request.app.state, "source_repository", None)
    document_store = request.app.state.document_store

    try:
        if source_repo:
            source_entry = await source_repo.get_source(source_id)
            if source_entry:
                return SourceDetailResponse(**source_entry)

        # Fallback to document lookup by chunk ID
        document = await document_store.get_by_id(source_id)
        if not document:
            raise HTTPException(status_code=404, detail=f"Source {source_id} not found")

        metadata = document.metadata.model_dump() if hasattr(document.metadata, 'model_dump') else document.metadata
        summary = SourceDetailResponse(
            source_id=metadata.get("source_id", source_id),
            title=metadata.get("title"),
            canonical_url=metadata.get("canonical_url") or metadata.get("source"),
            reference_path=metadata.get("reference_path") or metadata.get("section"),
            document_type=str(metadata.get("type")) if metadata.get("type") else None,
            section=metadata.get("section"),
            chunk_count=1,
            document_count=1,
            last_ingested_at=metadata.get("ingested_at"),
            metadata=metadata,
            chunks=[{
                "chunk_id": document.id,
                "document_id": document.parent_id,
                "ingested_at": metadata.get("ingested_at")
            }]
        )
        return summary

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Source retrieval failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/queries/{query_id}/sources", response_model=List[QuerySourceResponse])
async def get_sources_for_query(
    request: Request,
    query_id: str
) -> List[QuerySourceResponse]:
    """Return the sources cited in a previously logged query."""
    source_repo = getattr(request.app.state, "source_repository", None)
    if not source_repo:
        raise HTTPException(status_code=503, detail="Source repository unavailable")

    try:
        records = await source_repo.get_sources_for_query(query_id)
        return [QuerySourceResponse(**record) for record in records]
    except Exception as exc:
        logger.error("Failed to fetch query sources: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
