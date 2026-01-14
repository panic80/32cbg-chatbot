"""Cache management endpoints for admin API."""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app.api.admin.dependencies import get_cache_service, get_container
from app.api.security import verify_admin_bearer_token
from app.core.logging import get_logger
from app.services.cache import CacheService

logger = get_logger(__name__)
router = APIRouter(tags=["cache"])


class CacheManagementRequest(BaseModel):
    """Cache management request."""
    action: str = Field(..., description="Action: clear, warm, stats")
    patterns: Optional[List[str]] = Field(None, description="Patterns to clear (for clear action)")
    warm_queries: Optional[List[str]] = Field(None, description="Queries to warm cache with")


@router.post("/cache/manage")
async def manage_cache(
    cache_request: CacheManagementRequest,
    request: Request,
    _: bool = Depends(verify_admin_bearer_token),
    cache_service: CacheService = Depends(get_cache_service),
) -> Dict[str, Any]:
    """Manage cache operations."""
    try:
        if cache_request.action == "clear":
            if cache_request.patterns:
                cleared = 0
                for pattern in cache_request.patterns:
                    keys = await cache_service.redis.keys(pattern)
                    if keys:
                        await cache_service.redis.delete(*keys)
                        cleared += len(keys)
                return {
                    "status": "success",
                    "action": "clear",
                    "cleared_keys": cleared,
                }
            else:
                await cache_service.redis.flushdb()
                return {
                    "status": "success",
                    "action": "clear",
                    "message": "All cache cleared",
                }

        elif cache_request.action == "warm":
            if not cache_request.warm_queries:
                cache_request.warm_queries = [
                    "What is the meal allowance?",
                    "What is the POMV rate?",
                    "How do I claim travel expenses?",
                    "What documentation do I need?",
                ]

            from app.pipelines.improved_retrieval import ImprovedRetrievalPipeline

            container = request.app.state.container

            pipeline = ImprovedRetrievalPipeline(
                document_store=container.document_store,
                vector_store=container.vector_store_manager,
                cache_service=cache_service,
            )

            warmed = 0
            for query in cache_request.warm_queries:
                try:
                    await pipeline.retrieve(query)
                    warmed += 1
                except Exception as e:
                    logger.error(f"Failed to warm cache for query '{query}': {e}")

            return {
                "status": "success",
                "action": "warm",
                "warmed_queries": warmed,
            }

        elif cache_request.action == "stats":
            info = await cache_service.redis.info()

            patterns = {
                "retrieval:*": len(await cache_service.redis.keys("retrieval:*")),
                "llm:*": len(await cache_service.redis.keys("llm:*")),
                "embedding:*": len(await cache_service.redis.keys("embedding:*")),
            }

            return {
                "status": "success",
                "action": "stats",
                "stats": {
                    "used_memory_mb": info.get("used_memory", 0) / 1024 / 1024,
                    "total_keys": info.get("db0", {}).get("keys", 0),
                    "hit_rate": info.get("keyspace_hits", 0)
                    / (info.get("keyspace_hits", 0) + info.get("keyspace_misses", 1)),
                    "patterns": patterns,
                },
            }

        else:
            raise HTTPException(
                status_code=400, detail=f"Unknown action: {cache_request.action}"
            )

    except Exception as e:
        logger.error(f"Cache management failed: {e}")
        raise HTTPException(status_code=500, detail="Cache management failed")
