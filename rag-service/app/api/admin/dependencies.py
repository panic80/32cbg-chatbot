"""Shared dependencies for admin API endpoints."""

from fastapi import Request

from app.services.document_store import DocumentStore
from app.services.cache import CacheService


def get_container(request: Request):
    """Get service container from app state."""
    return request.app.state.container


async def get_document_store(request: Request) -> DocumentStore:
    """Get document store instance from container."""
    return request.app.state.container.document_store


async def get_cache_service(request: Request) -> CacheService:
    """Get cache service instance from container."""
    return request.app.state.container.cache_service


async def get_vector_store(request: Request):
    """Get vector store manager instance from container."""
    return request.app.state.container.vector_store_manager
