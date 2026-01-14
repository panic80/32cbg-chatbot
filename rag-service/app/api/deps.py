from typing import Generator
from fastapi import Request, Depends
from app.core.container import ServiceContainer, get_container
from app.core.interfaces import (
    ICacheService,
    IVectorStoreManager,
    IDocumentStore,
    ISourceRepository,
    IQueryLogger,
)

def get_service_container(request: Request) -> ServiceContainer:
    """Get the service container."""
    # Try getting from app state first (lifespan managed)
    if hasattr(request.app.state, "container"):
        return request.app.state.container
    # Fallback to global (e.g. if testing without full app)
    return get_container()

def get_vector_store_manager(
    container: ServiceContainer = Depends(get_service_container)
) -> IVectorStoreManager:
    return container.vector_store_manager

def get_document_store(
    container: ServiceContainer = Depends(get_service_container)
) -> IDocumentStore:
    return container.document_store

def get_cache_service(
    container: ServiceContainer = Depends(get_service_container)
) -> ICacheService:
    return container.cache_service

def get_source_repository(
    container: ServiceContainer = Depends(get_service_container)
) -> ISourceRepository:
    return container.source_repository

def get_query_logger(
    container: ServiceContainer = Depends(get_service_container)
) -> IQueryLogger:
    return container.query_logger
