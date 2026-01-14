"""Metrics export endpoints for admin API."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse

from app.api.security import verify_admin_bearer_token
from app.core.logging import get_logger
from app.services.performance_monitor import performance_monitor

logger = get_logger(__name__)
router = APIRouter(tags=["metrics"])


@router.get("/metrics/export")
async def export_metrics(
    format: str = "prometheus",
    _: bool = Depends(verify_admin_bearer_token),
) -> Any:
    """Export performance metrics."""
    try:
        metrics_data = await performance_monitor.export_metrics(format)

        if format == "prometheus":
            return PlainTextResponse(content=metrics_data)
        else:
            return metrics_data

    except Exception as e:
        logger.error(f"Failed to export metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to export metrics")
