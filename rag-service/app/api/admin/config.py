"""Configuration management endpoints for admin API."""

import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
import aiofiles

from app.api.security import verify_admin_bearer_token
from app.core.config import settings
from app.core.logging import get_logger
from app.services.model_selector import reload_model_selector

logger = get_logger(__name__)
router = APIRouter(tags=["config"])


class ConfigUpdateRequest(BaseModel):
    """Configuration update request."""
    config_updates: Dict[str, Any] = Field(
        ..., description="Configuration key-value pairs to update"
    )
    restart_required: bool = Field(
        False, description="Whether service restart is required"
    )


@router.get("/config/status")
async def get_configuration_status(
    _: bool = Depends(verify_admin_bearer_token),
) -> Dict[str, Any]:
    """Get current configuration status for hot-toggleable settings."""
    try:
        return {
            "status": "success",
            "config": {
                "enable_hyde": settings.enable_hyde,
                "hyde_model": settings.hyde_model,
                "hyde_timeout": settings.hyde_timeout,
                "enable_query_logging": getattr(settings, "enable_query_logging", True),
            },
        }
    except Exception as e:
        logger.error(f"Failed to get configuration status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get configuration status")


@router.post("/config/update")
async def update_configuration(
    request: ConfigUpdateRequest,
    _: bool = Depends(verify_admin_bearer_token),
) -> Dict[str, Any]:
    """Update system configuration (hot reload where possible)."""
    try:
        updated = []
        require_restart = []

        for key, value in request.config_updates.items():
            if hasattr(settings, key):
                old_value = getattr(settings, key)
                setattr(settings, key, value)
                updated.append(f"{key}: {old_value} -> {value}")

                # Check if restart required
                if key in ["VECTOR_STORE_TYPE", "EMBEDDING_MODEL", "DATABASE_URL"]:
                    require_restart.append(key)
            else:
                logger.warning(f"Unknown configuration key: {key}")

        # Write to .env file if specified
        if os.getenv("PERSIST_CONFIG_UPDATES", "false").lower() == "true":
            env_file = ".env"
            if os.path.exists(env_file):
                with open(env_file, "r") as f:
                    lines = f.readlines()

                for key, value in request.config_updates.items():
                    found = False
                    for i, line in enumerate(lines):
                        if line.startswith(f"{key}="):
                            lines[i] = f"{key}={value}\n"
                            found = True
                            break
                    if not found:
                        lines.append(f"{key}={value}\n")

                with open(env_file, "w") as f:
                    f.writelines(lines)

        return {
            "status": "success",
            "updated": updated,
            "restart_required": len(require_restart) > 0,
            "restart_required_for": require_restart,
        }

    except Exception as e:
        logger.error(f"Configuration update failed: {e}")
        raise HTTPException(status_code=500, detail="Configuration update failed")


@router.post("/config/reload-models")
async def reload_model_config(
    _: bool = Depends(verify_admin_bearer_token),
) -> Dict[str, Any]:
    """Reload model configuration from disk."""
    try:
        reload_model_selector()
        return {"status": "success", "message": "Model configuration reloaded"}
    except Exception as e:
        logger.error(f"Failed to reload model config: {e}")
        raise HTTPException(status_code=500, detail="Failed to reload model config")


@router.get("/logs/tail")
async def tail_logs(
    lines: int = 100,
    level: Optional[str] = None,
    _: bool = Depends(verify_admin_bearer_token),
) -> List[Dict[str, Any]]:
    """Get recent log entries."""
    try:
        import json

        log_file = "./logs/rag_service.log"
        if not os.path.exists(log_file):
            return []

        logs = []

        async with aiofiles.open(log_file, "r") as f:
            await f.seek(0, 2)
            file_size = await f.tell()

            chunk_size = min(file_size, lines * 500)
            await f.seek(max(0, file_size - chunk_size))

            content = await f.read()
            log_lines = content.strip().split("\n")

            for line in log_lines[-lines:]:
                try:
                    log_entry = json.loads(line)
                    if not level or log_entry.get("level") == level.upper():
                        logs.append(log_entry)
                except:
                    pass

        return logs

    except Exception as e:
        logger.error(f"Failed to read logs: {e}")
        raise HTTPException(status_code=500, detail="Failed to read logs")
