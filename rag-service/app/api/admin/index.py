"""Index management endpoints for admin API."""

import glob
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.admin.dependencies import get_document_store, get_vector_store, get_cache_service
from app.api.security import verify_admin_bearer_token
from app.core.logging import get_logger
from app.core.vectorstore import VectorStore
from app.pipelines.ingestion import IngestionPipeline
from app.services.document_store import DocumentStore
from app.services.cache import CacheService

logger = get_logger(__name__)
router = APIRouter(tags=["index"])


class IndexRebuildRequest(BaseModel):
    """Request to rebuild index."""
    clear_existing: bool = Field(True, description="Clear existing index before rebuild")
    source_directory: Optional[str] = Field(None, description="Directory to ingest from")
    file_patterns: List[str] = Field(
        default_factory=lambda: ["*.pdf", "*.txt", "*.md", "*.csv"],
        description="File patterns to include",
    )


@router.post("/index/rebuild")
async def rebuild_index(
    request: IndexRebuildRequest,
    background_tasks: BackgroundTasks,
    _: bool = Depends(verify_admin_bearer_token),
    document_store: DocumentStore = Depends(get_document_store),
    vector_store: VectorStore = Depends(get_vector_store),
    cache_service: CacheService = Depends(get_cache_service),
) -> Dict[str, Any]:
    """Rebuild the entire index."""
    try:
        async def rebuild_task():
            try:
                if request.clear_existing:
                    logger.info("Clearing existing index...")
                    await document_store.clear()

                pipeline = IngestionPipeline(
                    vector_store_manager=document_store.vector_store,
                    cache_service=cache_service,
                    source_repository=getattr(document_store, "source_repository", None),
                )

                source_dir = request.source_directory or "./documents"
                logger.info(f"Starting ingestion from {source_dir}")

                files_to_ingest = []
                for pattern in request.file_patterns:
                    files = glob.glob(f"{source_dir}/**/{pattern}", recursive=True)
                    files_to_ingest.extend(files)

                for file_path in files_to_ingest:
                    try:
                        await pipeline.ingest_file(file_path)
                    except Exception as e:
                        logger.error(f"Failed to ingest {file_path}: {e}")

                logger.info(f"Index rebuild completed. Ingested {len(files_to_ingest)} files")

            except Exception as e:
                logger.error(f"Index rebuild failed: {e}")

        background_tasks.add_task(rebuild_task)

        return {
            "status": "started",
            "message": "Index rebuild started in background",
            "clear_existing": request.clear_existing,
            "source_directory": request.source_directory,
        }

    except Exception as e:
        logger.error(f"Failed to start index rebuild: {e}")
        raise HTTPException(status_code=500, detail="Failed to start index rebuild")
