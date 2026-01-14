"""Health check endpoints for admin API."""

import time
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.api.admin.dependencies import (
    get_document_store,
    get_cache_service,
    get_vector_store,
)
from app.api.security import verify_admin_bearer_token
from app.core.config import settings
from app.core.logging import get_logger
from app.services.document_store import DocumentStore
from app.services.cache import CacheService
from app.services.performance_monitor import performance_monitor

logger = get_logger(__name__)
router = APIRouter(tags=["health"])

# Track startup time
startup_time = time.time()


class SystemStatus(BaseModel):
    """System health status."""
    status: str
    uptime_seconds: float
    version: str
    components: Dict[str, Dict[str, Any]]
    active_connections: int
    memory_usage_mb: float
    cpu_percent: float


@router.get("/health", response_model=SystemStatus)
async def get_system_health(
    _: bool = Depends(verify_admin_bearer_token),
    document_store: DocumentStore = Depends(get_document_store),
    cache_service: CacheService = Depends(get_cache_service),
    vector_store=Depends(get_vector_store),
) -> SystemStatus:
    """Get comprehensive system health status."""
    try:
        metrics = await performance_monitor.get_metrics_summary()

        components = {
            "document_store": {
                "status": "healthy",
                "document_count": await document_store.get_stats(),
            },
            "cache": {
                "status": "healthy" if cache_service.is_connected else "unhealthy",
                "hit_rate": metrics["cache"]["hit_rate"],
                "total_requests": metrics["cache"]["requests"],
            },
            "vector_store": {
                "status": "healthy",
                "type": settings.VECTOR_STORE_TYPE,
                "embedding_model": settings.EMBEDDING_MODEL,
            },
            "llm_pool": {
                "status": "healthy",
                "active_connections": metrics["system"]["active_connections"],
            },
        }

        return SystemStatus(
            status=(
                "healthy"
                if all(c["status"] == "healthy" for c in components.values())
                else "degraded"
            ),
            uptime_seconds=time.time() - startup_time,
            version="1.0.0",
            components=components,
            active_connections=metrics["system"]["active_connections"],
            memory_usage_mb=metrics["system"]["memory_mb"],
            cpu_percent=metrics["system"]["cpu_percent"],
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")
