"""Application factory."""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import os
from typing import Dict, Any

from app.core.config import settings
from app.core.logging import setup_logging, get_logger
from app.core.lifecycle import lifespan
from app.api import health, chat, ingestion, sources, websocket, progress, streaming_chat, admin, metrics

# Set up logging (should be done early)
setup_logging(settings.log_level, settings.log_format)
logger = get_logger(__name__)

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    logger.info(f"[DIAGNOSTIC] Creating FastAPI app with api_prefix: {settings.api_prefix}")
    
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        lifespan=lifespan,
        docs_url=f"{settings.api_prefix}/docs",
        redoc_url=f"{settings.api_prefix}/redoc",
        openapi_url=f"{settings.api_prefix}/openapi.json",
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add request timing middleware
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response

    # Exception handlers
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        logger.error(f"Validation error: {exc}")
        return JSONResponse(
            status_code=400,
            content={"error": "Bad Request", "message": str(exc)},
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal Server Error", "message": "An unexpected error occurred"},
        )

    # Include routers
    app.include_router(health.router, prefix=settings.api_prefix, tags=["health"])
    app.include_router(chat.router, prefix=settings.api_prefix, tags=["chat"])
    app.include_router(ingestion.router, prefix=settings.api_prefix, tags=["ingestion"])
    app.include_router(sources.router, prefix=settings.api_prefix, tags=["sources"])
    app.include_router(websocket.router, prefix=settings.api_prefix, tags=["websocket"])
    app.include_router(progress.router, prefix=settings.api_prefix, tags=["progress"])
    app.include_router(streaming_chat.router, prefix=settings.api_prefix, tags=["streaming"])
    app.include_router(metrics.router, prefix=settings.api_prefix, tags=["metrics"])
    app.include_router(admin.router, prefix=f"{settings.api_prefix}/admin", tags=["admin"])

    @app.get("/")
    async def root() -> Dict[str, Any]:
        """Root endpoint."""
        return {
            "service": settings.app_name,
            "version": settings.app_version,
            "status": "operational",
            "docs": f"{settings.api_prefix}/docs",
        }

    return app
