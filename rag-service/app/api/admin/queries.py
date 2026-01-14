"""Query history and statistics endpoints for admin API."""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response

from app.api.security import verify_admin_bearer_token
from app.core.logging import get_logger
from app.models.query_history import (
    QueryHistoryEntry,
    QueryHistoryFilter,
    QueryStatistics,
    QueryExportRequest,
)
from app.services.query_logger import get_query_logger

logger = get_logger(__name__)
router = APIRouter(tags=["queries"])


@router.post("/queries/history", response_model=List[QueryHistoryEntry])
async def get_query_history(
    filters: QueryHistoryFilter,
    _: bool = Depends(verify_admin_bearer_token),
) -> List[QueryHistoryEntry]:
    """Get query history with filtering and pagination."""
    try:
        query_logger = get_query_logger()
        history = await query_logger.get_query_history(filters)
        return history

    except Exception as e:
        logger.error(f"Failed to get query history: {e}")
        raise HTTPException(status_code=500, detail="Failed to get query history")


@router.get("/queries/stats", response_model=QueryStatistics)
async def get_query_statistics(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    _: bool = Depends(verify_admin_bearer_token),
) -> QueryStatistics:
    """Get aggregated query statistics."""
    try:
        query_logger = get_query_logger()
        stats = await query_logger.get_statistics(start_date, end_date)
        return stats

    except Exception as e:
        logger.error(f"Failed to get query statistics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get query statistics")


@router.delete("/queries/clear")
async def clear_old_queries(
    days: int = Query(90, description="Delete queries older than this many days"),
    _: bool = Depends(verify_admin_bearer_token),
) -> Dict[str, Any]:
    """Clear old queries from history."""
    try:
        query_logger = get_query_logger()
        deleted_count = await query_logger.cleanup_old_queries(days)

        return {
            "status": "success",
            "deleted_count": deleted_count,
            "message": f"Deleted {deleted_count} queries older than {days} days",
        }

    except Exception as e:
        logger.error(f"Failed to clear old queries: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear old queries")


@router.post("/queries/export")
async def export_queries(
    export_request: QueryExportRequest,
    _: bool = Depends(verify_admin_bearer_token),
):
    """Export query history to CSV or JSON format."""
    try:
        query_logger = get_query_logger()
        export_data = await query_logger.export_queries(export_request)

        if export_request.format == "csv":
            media_type = "text/csv"
            filename = f"query_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        else:
            media_type = "application/json"
            filename = f"query_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        headers = {"Content-Disposition": f"attachment; filename={filename}"}

        return Response(content=export_data, media_type=media_type, headers=headers)

    except Exception as e:
        logger.error(f"Failed to export queries: {e}")
        raise HTTPException(status_code=500, detail="Failed to export queries")


@router.get("/queries/realtime")
async def get_realtime_queries(
    minutes: int = Query(5, description="Get queries from last N minutes"),
    _: bool = Depends(verify_admin_bearer_token),
) -> List[QueryHistoryEntry]:
    """Get real-time query activity from the last N minutes."""
    try:
        query_logger = get_query_logger()

        start_date = datetime.utcnow() - timedelta(minutes=minutes)
        filters = QueryHistoryFilter(
            start_date=start_date,
            limit=100,
            order_by="timestamp",
            order_desc=True,
        )

        history = await query_logger.get_query_history(filters)
        return history

    except Exception as e:
        logger.error(f"Failed to get realtime queries: {e}")
        raise HTTPException(status_code=500, detail="Failed to get realtime queries")
