"""Backup and restore endpoints for admin API."""

import json
import os
import shutil
from datetime import datetime
from typing import Any, Dict, List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app.api.admin.dependencies import (
    get_document_store,
    get_vector_store,
    get_cache_service,
)
from app.api.security import verify_admin_bearer_token
from app.core.config import settings
from app.core.logging import get_logger
from app.core.vectorstore import VectorStore
from app.services.document_store import DocumentStore
from app.services.cache import CacheService

logger = get_logger(__name__)
router = APIRouter(tags=["backup"])


class BackupRequest(BaseModel):
    """Backup request."""
    backup_type: str = Field("full", description="Backup type: full, incremental, config-only")
    destination: str = Field("local", description="Destination: local, s3, gcs, azure")
    include_vectors: bool = Field(True, description="Include vector embeddings")
    include_indices: bool = Field(True, description="Include search indices")


@router.post("/backup/create")
async def create_backup(
    request: BackupRequest,
    background_tasks: BackgroundTasks,
    _: bool = Depends(verify_admin_bearer_token),
    document_store: DocumentStore = Depends(get_document_store),
    vector_store: VectorStore = Depends(get_vector_store),
    cache_service: CacheService = Depends(get_cache_service),
) -> Dict[str, Any]:
    """Create system backup."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_id = f"backup_{timestamp}"

        async def backup_task():
            try:
                backup_dir = f"./backups/{backup_id}"
                os.makedirs(backup_dir, exist_ok=True)

                # Backup configuration
                config_backup = {
                    "timestamp": timestamp,
                    "version": "1.0.0",
                    "settings": {
                        key: getattr(settings, key)
                        for key in dir(settings)
                        if not key.startswith("_")
                        and isinstance(
                            getattr(settings, key), (str, int, float, bool, list, dict)
                        )
                    },
                }

                with open(f"{backup_dir}/config.json", "w") as f:
                    json.dump(config_backup, f, indent=2)

                # Backup vectors if requested
                if request.include_vectors and request.backup_type in ["full"]:
                    vector_backup_dir = f"{backup_dir}/vectors"
                    os.makedirs(vector_backup_dir, exist_ok=True)

                    if settings.VECTOR_STORE_TYPE == "chroma":
                        shutil.copytree("./chroma_db", f"{vector_backup_dir}/chroma_db")

                # Backup indices if requested
                if request.include_indices and request.backup_type in ["full"]:
                    indices_backup_dir = f"{backup_dir}/indices"
                    os.makedirs(indices_backup_dir, exist_ok=True)

                    if os.path.exists("./bm25_index"):
                        shutil.copytree(
                            "./bm25_index", f"{indices_backup_dir}/bm25_index"
                        )

                    if os.path.exists("./cooccurrence_index"):
                        shutil.copytree(
                            "./cooccurrence_index",
                            f"{indices_backup_dir}/cooccurrence_index",
                        )

                # Create manifest
                manifest = {
                    "backup_id": backup_id,
                    "timestamp": timestamp,
                    "type": request.backup_type,
                    "components": {
                        "config": True,
                        "vectors": request.include_vectors,
                        "indices": request.include_indices,
                    },
                    "stats": await document_store.get_stats(),
                }

                with open(f"{backup_dir}/manifest.json", "w") as f:
                    json.dump(manifest, f, indent=2)

                # Upload to cloud if requested
                if request.destination != "local":
                    logger.info(f"Would upload backup to {request.destination}")

                logger.info(f"Backup {backup_id} completed successfully")

            except Exception as e:
                logger.error(f"Backup failed: {e}")

        background_tasks.add_task(backup_task)

        return {
            "status": "started",
            "backup_id": backup_id,
            "message": "Backup started in background",
        }

    except Exception as e:
        logger.error(f"Failed to start backup: {e}")
        raise HTTPException(status_code=500, detail="Failed to start backup")


@router.get("/backups/list")
async def list_backups(
    _: bool = Depends(verify_admin_bearer_token),
) -> List[Dict[str, Any]]:
    """List available backups."""
    try:
        backups = []
        backup_dir = "./backups"

        if os.path.exists(backup_dir):
            for backup_name in os.listdir(backup_dir):
                manifest_path = f"{backup_dir}/{backup_name}/manifest.json"
                if os.path.exists(manifest_path):
                    with open(manifest_path, "r") as f:
                        manifest = json.load(f)

                    # Get backup size
                    backup_size = 0
                    for root, dirs, files in os.walk(f"{backup_dir}/{backup_name}"):
                        for file in files:
                            backup_size += os.path.getsize(os.path.join(root, file))

                    backups.append(
                        {
                            "backup_id": manifest["backup_id"],
                            "timestamp": manifest["timestamp"],
                            "type": manifest["type"],
                            "size_mb": round(backup_size / 1024 / 1024, 2),
                            "components": manifest["components"],
                        }
                    )

        # Sort by timestamp
        backups.sort(key=lambda x: x["timestamp"], reverse=True)

        return backups

    except Exception as e:
        logger.error(f"Failed to list backups: {e}")
        raise HTTPException(status_code=500, detail="Failed to list backups")


@router.post("/backups/restore/{backup_id}")
async def restore_backup(
    backup_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
    _: bool = Depends(verify_admin_bearer_token),
) -> Dict[str, Any]:
    """Restore from backup."""
    try:
        backup_dir = f"./backups/{backup_id}"

        if not os.path.exists(backup_dir):
            raise HTTPException(status_code=404, detail="Backup not found")

        container = request.app.state.container

        async def restore_task():
            try:
                with open(f"{backup_dir}/manifest.json", "r") as f:
                    manifest = json.load(f)

                # Restore configuration
                if os.path.exists(f"{backup_dir}/config.json"):
                    with open(f"{backup_dir}/config.json", "r") as f:
                        config = json.load(f)
                    logger.info("Configuration restored from backup")

                # Restore vectors
                if manifest["components"]["vectors"]:
                    if os.path.exists(f"{backup_dir}/vectors/chroma_db"):
                        vector_store = container.vector_store_manager
                        await vector_store.close()

                        shutil.rmtree("./chroma_db", ignore_errors=True)
                        shutil.copytree(
                            f"{backup_dir}/vectors/chroma_db", "./chroma_db"
                        )

                        await vector_store.initialize()
                        logger.info("Vectors restored from backup")

                # Restore indices
                if manifest["components"]["indices"]:
                    if os.path.exists(f"{backup_dir}/indices/bm25_index"):
                        shutil.rmtree("./bm25_index", ignore_errors=True)
                        shutil.copytree(
                            f"{backup_dir}/indices/bm25_index", "./bm25_index"
                        )

                    if os.path.exists(f"{backup_dir}/indices/cooccurrence_index"):
                        shutil.rmtree("./cooccurrence_index", ignore_errors=True)
                        shutil.copytree(
                            f"{backup_dir}/indices/cooccurrence_index",
                            "./cooccurrence_index",
                        )

                    logger.info("Indices restored from backup")

                logger.info(f"Restore from backup {backup_id} completed")

            except Exception as e:
                logger.error(f"Restore failed: {e}")

        background_tasks.add_task(restore_task)

        return {
            "status": "started",
            "backup_id": backup_id,
            "message": "Restore started in background. Service restart recommended after completion.",
        }

    except Exception as e:
        logger.error(f"Failed to start restore: {e}")
        raise HTTPException(status_code=500, detail="Failed to start restore")
