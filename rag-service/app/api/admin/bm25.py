"""BM25 management endpoints for admin API."""

from typing import Dict, Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from app.api.security import verify_admin_bearer_token
from app.core.logging import get_logger
from app.services.bm25 import rebuild_bm25_index

logger = get_logger(__name__)
router = APIRouter(tags=["bm25"])


@router.post("/bm25/rebuild")
async def rebuild_bm25(
    background_tasks: BackgroundTasks,
    _: bool = Depends(verify_admin_bearer_token),
) -> Dict[str, Any]:
    """Rebuild the BM25 index in background."""
    try:
        async def rebuild_task():
            try:
                logger.info("Starting BM25 index rebuild...")
                await rebuild_bm25_index()
                logger.info("BM25 index rebuild completed")
            except Exception as e:
                logger.error(f"BM25 index rebuild failed: {e}")

        background_tasks.add_task(rebuild_task)

        return {
            "status": "started",
            "message": "BM25 index rebuild started in background",
        }

    except Exception as e:
        logger.error(f"Failed to start BM25 index rebuild: {e}")
        raise HTTPException(status_code=500, detail="Failed to start BM25 index rebuild")
