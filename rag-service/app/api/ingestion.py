"""Document ingestion API endpoints."""

from fastapi import APIRouter, Request, HTTPException, UploadFile, File, Form, Query, Depends
from typing import List, Optional
import os
import tempfile
from datetime import datetime, timezone
import uuid

from app.core.logging import get_logger
from app.models.documents import (
    DocumentIngestionRequest, DocumentIngestionResponse,
    DocumentType
)
from app.pipelines.ingestion import IngestionPipeline
from app.services.cache import CacheService
from app.api.websocket import progress_tracker
from app.api.progress import send_progress_update, close_progress_stream
from app.api.security import verify_admin_bearer_token

logger = get_logger(__name__)

router = APIRouter(dependencies=[Depends(verify_admin_bearer_token)])


@router.post("/database/purge")
async def purge_database(request: Request) -> dict:
    """Purge all documents from the vector database."""
    try:
        # Get services from app state
        app = request.app
        vector_store_manager = app.state.vector_store_manager
        cache_service = getattr(app.state, "cache_service", None)
        
        logger.info("Starting database purge...")
        
        # Clear the vector store collection
        if hasattr(vector_store_manager.vector_store, 'delete_collection'):
            # For Chroma
            vector_store_manager.vector_store.delete_collection()
            logger.info("Deleted vector store collection")
            
            # Recreate the collection
            vector_store_manager.vector_store = vector_store_manager._create_vector_store()
            logger.info("Recreated empty vector store collection")
        elif hasattr(vector_store_manager.vector_store, 'delete'):
            # For other vector stores that support delete without IDs
            vector_store_manager.vector_store.delete(delete_all=True)
            logger.info("Deleted all documents from vector store")
        else:
            raise HTTPException(
                status_code=501, 
                detail="Vector store does not support purge operation"
            )
        
        # Clear cache if available
        if cache_service:
            await cache_service.clear_all()
            logger.info("Cleared all cache entries")

        # Clear source repository if available
        source_repository = getattr(app.state, "source_repository", None)
        if source_repository:
            await source_repository.clear_all()
            logger.info("Cleared source repository")

        # Get updated stats
        document_count = 0
        if hasattr(vector_store_manager.vector_store, '_collection'):
            # For Chroma
            collection = vector_store_manager.vector_store._collection
            document_count = collection.count() if collection else 0

        # Invalidate retrieval pipeline cache and BM25 corpus cache
        try:
            # Refresh the BM25 corpus cache (now empty)
            vector_store_manager.get_all_documents(refresh=True)
            if hasattr(app.state, 'retrieval_pipeline_cache'):
                app.state.retrieval_pipeline_cache.clear()
                logger.info("Cleared retrieval pipeline cache after purge")
        except Exception as e:
            logger.warning(f"Failed to refresh caches after purge: {e}")
        
        logger.info(f"Database purge completed. Document count: {document_count}")
        
        return {
            "status": "success",
            "message": "Database purged successfully",
            "document_count": document_count,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Database purge failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest", response_model=DocumentIngestionResponse)
async def ingest_document(
    request: Request,
    ingestion_request: DocumentIngestionRequest
) -> DocumentIngestionResponse:
    """Ingest a document from URL or direct content."""
    try:
        # Get services from app state
        app = request.app
        vector_store = app.state.vector_store_manager
        cache_service = getattr(app.state, "cache_service", None)
        source_repository = getattr(app.state, "source_repository", None)
        
        # Create ingestion pipeline
        pipeline = IngestionPipeline(
            vector_store,
            cache_service,
            source_repository=source_repository
        )
        
        # Create operation ID based on URL or content hash
        if ingestion_request.url:
            operation_id = f"url_{hash(ingestion_request.url)}"
        else:
            operation_id = f"ingest_{int(datetime.now(timezone.utc).timestamp())}"
        
        # Store URL mapping for progress tracking
        if ingestion_request.url:
            # Store URL -> operation_id mapping (in-memory for now)
            if not hasattr(app.state, 'url_operations'):
                app.state.url_operations = {}
            app.state.url_operations[ingestion_request.url] = operation_id
        
        async def progress_callback(event_type: str, data: dict):
            await send_progress_update(operation_id, event_type, data)
        
        # Ingest document with progress tracking
        response = await pipeline.ingest(
            ingestion_request,
            progress_callback=progress_callback
        )
        
        # Close progress stream
        await close_progress_stream(operation_id)
        
        # Refresh BM25 corpus and clear retrieval pipeline cache so new data is used
        try:
            vector_store.get_all_documents(refresh=True)
            if hasattr(app.state, 'retrieval_pipeline_cache'):
                app.state.retrieval_pipeline_cache.clear()
                logger.info("Cleared retrieval pipeline cache after ingestion")
        except Exception as e:
            logger.warning(f"Failed to refresh caches after ingestion: {e}")
        
        return response
        
    except Exception as e:
        logger.error(f"Document ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest/file", response_model=DocumentIngestionResponse)
async def ingest_file(
    request: Request,
    file: UploadFile = File(...),
    document_type: Optional[str] = Form(None),
    force_refresh: bool = Form(False),
    metadata: Optional[str] = Form("{}")
) -> DocumentIngestionResponse:
    """Ingest a document from file upload."""
    try:
        # Validate file type
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        # Determine document type
        if document_type:
            doc_type = DocumentType(document_type)
        else:
            # Auto-detect from extension
            if file_extension == ".pdf":
                doc_type = DocumentType.PDF
            elif file_extension in [".txt", ".text"]:
                doc_type = DocumentType.TEXT
            elif file_extension in [".md", ".markdown"]:
                doc_type = DocumentType.MARKDOWN
            elif file_extension in [".docx", ".doc"]:
                doc_type = DocumentType.DOCX
            elif file_extension in [".xlsx", ".xls"]:
                doc_type = DocumentType.XLSX
            elif file_extension == ".csv":
                doc_type = DocumentType.CSV
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
                
        # Save uploaded file temporarily
        content = await file.read()
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
        tmp_file.write(content)
        tmp_file.flush()
        tmp_file.close()  # Explicitly close file handle before passing to loader
        tmp_file_path = tmp_file.name
        
        try:
            # Parse metadata
            import json
            metadata_dict = json.loads(metadata)
            metadata_dict["original_filename"] = file.filename
            
            # Create ingestion request
            ingestion_request = DocumentIngestionRequest(
                file_path=tmp_file_path,
                type=doc_type,
                metadata=metadata_dict,
                force_refresh=force_refresh
            )
            
            # Get services
            app = request.app
            vector_store = app.state.vector_store_manager
            cache_service = getattr(app.state, "cache_service", None)
            source_repository = getattr(app.state, "source_repository", None)
            
            # Create pipeline and ingest
            pipeline = IngestionPipeline(
                vector_store,
                cache_service,
                source_repository=source_repository
            )
            response = await pipeline.ingest(ingestion_request)
            
            # Refresh BM25 corpus and clear retrieval pipeline cache
            try:
                vector_store.get_all_documents(refresh=True)
                if hasattr(app.state, 'retrieval_pipeline_cache'):
                    app.state.retrieval_pipeline_cache.clear()
                    logger.info("Cleared retrieval pipeline cache after file ingestion")
            except Exception as e:
                logger.warning(f"Failed to refresh caches after file ingestion: {e}")
            
            return response
            
        finally:
            # Clean up temp file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
                
    except Exception as e:
        logger.error(f"File ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest/batch", response_model=List[DocumentIngestionResponse])
async def ingest_batch(
    request: Request,
    ingestion_requests: List[DocumentIngestionRequest],
    max_concurrent: int = 5,
    client_id: Optional[str] = Query(None, description="WebSocket client ID for progress tracking")
) -> List[DocumentIngestionResponse]:
    """Ingest multiple documents concurrently with optional progress tracking.
    
    Args:
        ingestion_requests: List of documents to ingest
        max_concurrent: Maximum number of concurrent ingestions (default: 5, max: 10)
        client_id: Optional WebSocket client ID for real-time progress updates
    """
    try:
        # Limit max concurrent to prevent resource exhaustion
        max_concurrent = min(max_concurrent, 10)
        
        # Generate task ID for progress tracking
        task_id = str(uuid.uuid4())
        
        # Start progress tracking if client ID provided
        if client_id:
            await progress_tracker.start_task(
                task_id=task_id,
                client_id=client_id,
                total_items=len(ingestion_requests),
                task_type="batch_ingestion"
            )
        
        # Get services
        app = request.app
        vector_store = app.state.vector_store_manager
        cache_service = getattr(app.state, "cache_service", None)
        source_repository = getattr(app.state, "source_repository", None)
        
        # Create pipeline
        pipeline = IngestionPipeline(
            vector_store,
            cache_service,
            source_repository=source_repository
        )
        
        # Create progress callback
        async def progress_callback(index: int, total: int, response: DocumentIngestionResponse):
            if client_id:
                message = f"Processed document {index + 1}/{total}"
                if response.status == "error":
                    await progress_tracker.update_progress(
                        task_id=task_id,
                        increment=1,
                        message=message,
                        error=response.message
                    )
                else:
                    await progress_tracker.update_progress(
                        task_id=task_id,
                        increment=1,
                        message=message
                    )
        
        # Ingest all documents concurrently
        responses = await pipeline.ingest_batch(
            ingestion_requests,
            max_concurrent=max_concurrent,
            progress_callback=progress_callback if client_id else None
        )
        
        # Complete progress tracking
        if client_id:
            successful = sum(1 for r in responses if r.status == "success")
            failed = sum(1 for r in responses if r.status == "error")
            
            await progress_tracker.complete_task(
                task_id=task_id,
                success=failed == 0,
                message=f"Completed: {successful} successful, {failed} failed"
            )
        
        # Refresh BM25 corpus and clear retrieval pipeline cache
        try:
            vector_store.get_all_documents(refresh=True)
            if hasattr(app.state, 'retrieval_pipeline_cache'):
                app.state.retrieval_pipeline_cache.clear()
                logger.info("Cleared retrieval pipeline cache after batch ingestion")
        except Exception as e:
            logger.warning(f"Failed to refresh caches after batch ingestion: {e}")
        
        return responses
        
    except Exception as e:
        logger.error(f"Batch ingestion failed: {e}")
        
        # Mark task as failed
        if client_id and 'task_id' in locals():
            await progress_tracker.complete_task(
                task_id=task_id,
                success=False,
                message=str(e)
            )
            
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest/canada-ca", response_model=DocumentIngestionResponse)
async def ingest_canada_ca(request: Request) -> DocumentIngestionResponse:
    """Ingest all Canada.ca travel instructions."""
    try:
        # Get services
        app = request.app
        vector_store = app.state.vector_store_manager
        cache_service = getattr(app.state, "cache_service", None)
        source_repository = getattr(app.state, "source_repository", None)
        
        # Create pipeline
        pipeline = IngestionPipeline(
            vector_store,
            cache_service,
            source_repository=source_repository
        )
        
        # Run Canada.ca ingestion
        logger.info("Starting Canada.ca travel instructions ingestion")
        response = await pipeline.ingest_canada_ca()
        
        # Refresh BM25 corpus and clear retrieval pipeline cache
        try:
            vector_store.get_all_documents(refresh=True)
            if hasattr(app.state, 'retrieval_pipeline_cache'):
                app.state.retrieval_pipeline_cache.clear()
                logger.info("Cleared retrieval pipeline cache after Canada.ca ingestion")
        except Exception as e:
            logger.warning(f"Failed to refresh caches after Canada.ca ingestion: {e}")
        
        return response
        
    except Exception as e:
        logger.error(f"Canada.ca ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents/{document_id}")
async def delete_document(request: Request, document_id: str) -> dict:
    """Delete a document and all its chunks."""
    try:
        # Get document store
        app = request.app
        document_store = app.state.document_store
        
        # Delete document
        success = await document_store.delete_by_id(document_id)
        
        if success:
            return {
                "status": "success",
                "message": f"Document {document_id} deleted successfully"
            }
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Document {document_id} not found"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document deletion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
