"""Metrics API endpoints for RAG performance dashboard."""

from fastapi import APIRouter, HTTPException

from app.core.logging import get_logger
from app.services.performance_monitor import get_performance_monitor

logger = get_logger(__name__)

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("/summary")
async def metrics_summary() -> dict:
    """Return aggregated performance metrics for the dashboard."""
    monitor = get_performance_monitor()
    try:
        return monitor.get_dashboard_metrics()
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Failed to build metrics summary", exc_info=exc)
        raise HTTPException(status_code=500, detail="Failed to build metrics summary")


@router.get("/all")
async def metrics_all() -> dict:
    """Return all recorded metrics including detailed latencies."""
    monitor = get_performance_monitor()
    try:
        return monitor.get_metrics_summary()
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Failed to build full metrics", exc_info=exc)
        raise HTTPException(status_code=500, detail="Failed to build full metrics")
