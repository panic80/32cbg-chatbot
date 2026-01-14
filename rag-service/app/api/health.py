"""Health check endpoints."""

from fastapi import APIRouter, Request
from typing import Dict, Any
from datetime import datetime

from app.core.config import settings

router = APIRouter()


@router.get("/health")
async def health_check(request: Request) -> Dict[str, Any]:
    """Health check endpoint."""
    # Get app state
    app = request.app
    
    # Check vector store
    vector_store_status = "unknown"
    vector_store_stats = {}
    
    if hasattr(app.state, "vector_store_manager"):
        try:
            stats = await app.state.vector_store_manager.get_collection_stats()
            vector_store_status = "healthy"
            vector_store_stats = stats
        except Exception as e:
            vector_store_status = "unhealthy"
            vector_store_stats = {"error": str(e)}
            
    # Check cache
    cache_status = "disabled"
    if hasattr(app.state, "cache_service") and app.state.cache_service:
        try:
            # Simple ping test
            await app.state.cache_service.set("health_check", True, ttl=10)
            cache_status = "healthy"
        except Exception:
            cache_status = "unhealthy"
            
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "vector_store": {
                "status": vector_store_status,
                "type": settings.vector_store_type,
                **vector_store_stats
            },
            "cache": {
                "status": cache_status,
                "enabled": bool(settings.redis_url)
            },
            "embeddings": {
                "provider": "openai" if settings.openai_api_key else "google",
                "model": settings.openai_embedding_model if settings.openai_api_key else settings.google_embedding_model
            }
        }
    }


@router.get("/ready")
async def readiness_check(request: Request) -> Dict[str, str]:
    """Readiness check endpoint."""
    # Check if all services are initialized
    app = request.app
    
    if not hasattr(app.state, "vector_store_manager"):
        return {"status": "not_ready", "reason": "vector_store_not_initialized"}
        
    if not hasattr(app.state, "document_store"):
        return {"status": "not_ready", "reason": "document_store_not_initialized"}
        
    return {"status": "ready"}