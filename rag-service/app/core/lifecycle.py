"""Lifecycle management for the application."""
from contextlib import asynccontextmanager
from fastapi import FastAPI
import asyncio
from app.core.config import settings
from app.core.logging import get_logger
from app.core.container import ServiceContainer, set_container

logger = get_logger(__name__)

async def _preload_vector_store_corpus(container: ServiceContainer) -> None:
    """Preload vector store corpus in background."""
    try:
        await asyncio.to_thread(
            container.vector_store_manager.get_all_documents, True
        )
        logger.info("Vector store corpus preloaded for retrieval")
    except Exception as e:
        logger.warning(f"Vector store preload failed (non-fatal): {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    logger.info("Starting RAG service...")

    # Create and initialize service container
    container = ServiceContainer()

    try:
        await container.initialize(settings)

        # Set global container for module-level access
        set_container(container)

        # Store container in app state for route access
        app.state.container = container

        # Backward compatibility: set individual services in app.state
        app.state.document_store = container.document_store
        app.state.vector_store_manager = container.vector_store_manager
        app.state.cache_service = container.cache_service
        app.state.query_logger = container.query_logger
        app.state.source_repository = container.source_repository
        app.state.retrieval_pipeline_cache = container.retrieval_pipeline_cache

        # Start corpus preload as background task (non-blocking)
        preload_task = asyncio.create_task(
            _preload_vector_store_corpus(container)
        )
        app.state.preload_task = preload_task

        logger.info("RAG service started successfully")

    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

    yield

    # Cleanup
    logger.info("Shutting down RAG service...")

    # Cancel preload task if still running
    if hasattr(app.state, 'preload_task') and not app.state.preload_task.done():
        app.state.preload_task.cancel()
        try:
            await app.state.preload_task
        except asyncio.CancelledError:
            pass

    # Shutdown container
    await container.shutdown()

    logger.info("RAG service shut down")
