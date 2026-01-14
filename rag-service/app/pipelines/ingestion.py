"""Document ingestion pipeline."""

import uuid
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from datetime import datetime, timezone
import hashlib
import asyncio
from pathlib import Path

from langchain_core.documents import Document as LangchainDocument

from app.core.config import settings
from app.core.logging import get_logger
from app.core.vectorstore import VectorStoreManager
from app.core.errors import (
    IngestionError, NetworkError, ParsingError, 
    ValidationError, StorageError, categorize_error
)
from app.models.documents import (
    Document, DocumentType, DocumentMetadata,
    DocumentIngestionRequest, DocumentIngestionResponse
)
from app.pipelines.loaders import LangChainDocumentLoader, CanadaCaScraper
from app.pipelines.splitters import LangChainTextSplitter
from app.pipelines.smart_splitters import SmartDocumentSplitter
from app.pipelines.parallel_ingestion import (
    ParallelEmbeddingGenerator, ParallelChunkProcessor, OptimizedVectorStoreWriter
)
from app.services.cache import CacheService
from app.services.embedding_cache import EmbeddingCacheService
from app.services.progress_tracker import IngestionProgressTracker
from app.services.ingestion_checkpoint import (
    IngestionCheckpointService, CheckpointState
)
from app.services.metadata_extractor import MetadataExtractor
from app.services.quality_validator import ChunkQualityValidator
from app.utils.retry import retry_async, RetryManager, AGGRESSIVE_RETRY_CONFIG
from app.utils.deduplication import DeduplicationService, ContentHasher
from app.components.bm25_retriever import TravelBM25Retriever
from app.components.cooccurrence_indexer import CooccurrenceIndexer
from app.services.performance_monitor import get_performance_monitor
from app.models.source_catalog import SourceCatalogEntry

if TYPE_CHECKING:
    from app.services.source_repository import SourceRepository

logger = get_logger(__name__)


class IngestionPipeline:
    """Document ingestion pipeline."""
    
    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        cache_service: Optional[CacheService] = None,
        deduplication_threshold: float = 0.85,
        use_smart_chunking: bool = True,
        use_hierarchical_chunking: bool = False,
        llm: Optional[Any] = None,
        source_repository: Optional["SourceRepository"] = None
    ):
        """Initialize ingestion pipeline."""
        self.vector_store_manager = vector_store_manager
        self.cache_service = cache_service
        self.source_repository = source_repository
        self.performance_monitor = get_performance_monitor()
        self._active_ingestions = 0
        self.retry_manager = RetryManager(AGGRESSIVE_RETRY_CONFIG)
        self.deduplication_service = DeduplicationService(deduplication_threshold)
        self.content_hasher = ContentHasher()
        self.use_smart_chunking = use_smart_chunking
        self.use_hierarchical_chunking = use_hierarchical_chunking
        self.chunk_processor = ParallelChunkProcessor(max_workers=settings.parallel_chunk_workers)
        self.optimized_writer = None  # Initialize when needed
        self.progress_trackers: Dict[str, IngestionProgressTracker] = {}
        
        # Initialize embedding cache if cache service is available
        self.embedding_cache = None
        if cache_service and getattr(settings, 'enable_embedding_cache', True):
            self.embedding_cache = EmbeddingCacheService(
                cache_service, 
                ttl=settings.embedding_cache_ttl
            )
            logger.info("Embedding cache initialized")
            
        # Initialize checkpoint service
        self.checkpoint_service = None
        if cache_service:
            self.checkpoint_service = IngestionCheckpointService(cache_service)
            logger.info("Checkpoint service initialized")
        
        # Initialize table multi-vector retriever if LLM is provided
        self.table_retriever = None
        if llm and getattr(settings, 'enable_table_multivector', True):
            try:
                from app.components.table_multi_vector_retriever import TableMultiVectorRetriever
                self.table_retriever = TableMultiVectorRetriever(
                    vectorstore=vector_store_manager.vector_store,
                    llm=llm
                )
                logger.info("Table multi-vector retriever initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize table retriever: {e}")
                
        # Initialize metadata extractor and quality validator
        self.metadata_extractor = MetadataExtractor(llm=llm)
        self.quality_validator = ChunkQualityValidator(
            min_chunk_size=settings.min_chunk_size,
            max_chunk_size=settings.max_chunk_size,
            min_quality_score=getattr(settings, 'min_quality_score', 60.0)
        )
        logger.info("Metadata extractor and quality validator initialized")
        
        # Initialize co-occurrence indexer
        self.cooccurrence_indexer = CooccurrenceIndexer(
            index_path=Path("cooccurrence_index")
        )
        # Try to load existing index
        self.cooccurrence_indexer.load_index()
        
    async def cleanup(self):
        """Cleanup resources."""
        try:
            if self.chunk_processor:
                self.chunk_processor.close()
            if self.optimized_writer:
                self.optimized_writer.close()
            logger.info("Ingestion pipeline resources cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        
    async def ingest_document(
        self,
        request: DocumentIngestionRequest,
        progress_callback: Optional[callable] = None
    ) -> DocumentIngestionResponse:
        """Ingest a document through the pipeline."""
        start_time = datetime.now(timezone.utc)
        self.performance_monitor.adjust_ingestion_in_progress(+1)
        self.performance_monitor.increment_counter("ingestion_started", 1)
        invalid_chunk_count = 0
        deduplicated_docs: List[Document] = []

        # Create progress tracker
        operation_id = f"ingest_{int(start_time.timestamp())}"
        progress_tracker = IngestionProgressTracker(operation_id, request.url or "direct_input")
        if progress_callback:
            progress_tracker.add_callback(progress_callback)
        self.progress_trackers[operation_id] = progress_tracker
        
        try:
            # Validate request
            self._validate_request(request)
            
            # Check for resumable checkpoint
            checkpoint = None
            if self.checkpoint_service and hasattr(request, 'operation_id') and request.operation_id:
                checkpoint = await self.checkpoint_service.get_checkpoint(request.operation_id)
                if checkpoint and await self.checkpoint_service.can_resume(request.operation_id):
                    logger.info(f"Resuming operation {request.operation_id} from state: {checkpoint.current_state}")
                    operation_id = request.operation_id
                    progress_tracker = self.progress_trackers.get(operation_id, progress_tracker)
            
            # Create checkpoint for new operations
            if self.checkpoint_service and not checkpoint:
                checkpoint = await self.checkpoint_service.create_checkpoint(
                    operation_id=operation_id,
                    document_source=request.url or request.file_path or "direct_input",
                    total_documents=0  # Will be updated after loading
                )
            
            # Check if document already exists (unless force refresh or resuming)
            if not request.force_refresh and not checkpoint:
                existing = await self._check_existing_document(request)
                if existing:
                    processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
                    self.performance_monitor.record_ingestion_result(
                        processing_time_ms=processing_time * 1000,
                        chunks=existing.get("chunks", 0),
                        invalid_chunks=0,
                        status="skipped",
                    )
                    return DocumentIngestionResponse(
                        document_id=existing["id"],
                        chunks_created=existing["chunks"],
                        status="exists",
                        message="Document already ingested",
                        processing_time=0
                    )
                    
            # Load document with retry (skip if resuming from later stage)
            serialized_docs = []
            if checkpoint and checkpoint.metadata.get("loaded_documents"):
                serialized_docs = checkpoint.metadata.get("loaded_documents")
                documents = self._deserialize_documents_from_checkpoint(serialized_docs)
            if not checkpoint or checkpoint.current_state == CheckpointState.LOADING or not serialized_docs:
                logger.info(f"Loading document: {request.url or request.file_path}")
                await progress_tracker.start_step("loading")
                if self.checkpoint_service:
                    await self.checkpoint_service.update_progress(operation_id, new_state=CheckpointState.LOADING)
                documents = await self._load_documents_with_retry(request)
                await progress_tracker.complete_step("loading", f"Loaded {len(documents)} document(s)")
                
                # Update checkpoint with document count
                if self.checkpoint_service and checkpoint:
                    checkpoint.total_documents = len(documents)
                    await self.checkpoint_service.save_checkpoint(checkpoint)
                if self.checkpoint_service:
                    serialized_docs = self._serialize_documents_for_checkpoint(documents)
                    await self.checkpoint_service.update_progress(
                        operation_id,
                        metadata_update={"loaded_documents": serialized_docs}
                    )
            else:
                logger.info(
                    "Resuming from checkpoint state %s with cached documents",
                    checkpoint.current_state
                )

            if not documents:
                raise ParsingError("No content extracted from document")
                
            # Split documents into chunks (skip if resuming from later stage)
            cached_chunks = []
            if checkpoint and checkpoint.metadata.get("split_chunks"):
                cached_chunks = checkpoint.metadata.get("split_chunks")
                chunks = self._deserialize_chunks_from_checkpoint(cached_chunks)
            else:
                chunks = []

            if not chunks or not checkpoint or checkpoint.current_state in [CheckpointState.LOADING, CheckpointState.SPLITTING]:
                logger.info(f"Splitting {len(documents)} documents")
                await progress_tracker.start_step("splitting")
                if self.checkpoint_service:
                    await self.checkpoint_service.update_progress(operation_id, new_state=CheckpointState.SPLITTING)
                split_start = datetime.now(timezone.utc)
                chunks = await self._split_documents_safely(documents, progress_tracker)
                split_time = (datetime.now(timezone.utc) - split_start).total_seconds()
                await progress_tracker.complete_step("splitting", f"Created {len(chunks)} chunks in {split_time:.2f}s")
                logger.info(f"Document splitting completed in {split_time:.2f} seconds")
                if self.checkpoint_service:
                    serialized_chunks = self._serialize_chunks_for_checkpoint(chunks)
                    await self.checkpoint_service.update_progress(
                        operation_id,
                        metadata_update={"split_chunks": serialized_chunks}
                    )
            else:
                logger.info(
                    "Resuming with %s cached chunks from checkpoint state %s",
                    len(chunks),
                    checkpoint.current_state
                )
            
            if not chunks:
                raise ParsingError("No chunks created from document")

            # Extract metadata for chunks
            if getattr(settings, 'enable_metadata_extraction', True):
                logger.info("Extracting metadata for chunks")
                for chunk in chunks:
                    try:
                        enriched_metadata = await self.metadata_extractor.extract_metadata(chunk)
                        chunk.metadata.update(enriched_metadata)
                    except Exception as e:
                        logger.warning(f"Failed to extract metadata for chunk: {e}")

            # Extract column numbers from DOA documents
            logger.info("Extracting column numbers from DOA chunks")
            for chunk in chunks:
                try:
                    column_metadata = self._extract_column_number(chunk)
                    if column_metadata:
                        chunk.metadata.update(column_metadata)
                except Exception as e:
                    logger.warning(f"Failed to extract column number for chunk: {e}")

            # Add column headers to table chunks that lost them during PDF extraction
            logger.info("Adding column context to DOA table chunks")
            chunks = self._add_column_context_to_chunks(chunks, request)

            # Validate chunk quality
            if getattr(settings, 'enable_quality_validation', True):
                logger.info("Validating chunk quality")
                valid_chunks, invalid_chunks, quality_stats = self.quality_validator.validate_batch(chunks)
                invalid_chunk_count = len(invalid_chunks)
                logger.info(f"Quality validation: {quality_stats['valid_chunks']} valid, "
                          f"{quality_stats['invalid_chunks']} invalid, "
                          f"average score: {quality_stats['average_quality_score']}")

                # Use only valid chunks if validation is strict
                if getattr(settings, 'strict_quality_validation', False):
                    chunks = valid_chunks
                    if not chunks:
                        raise ParsingError("No chunks passed quality validation")
                else:
                    # Just log invalid chunks but keep all
                    if invalid_chunks:
                        logger.warning(f"{len(invalid_chunks)} chunks below quality threshold")

            # Generate document ID
            doc_id = self._generate_document_id(request)

            # Convert to internal document format
            internal_docs = self._convert_to_internal_documents(
                chunks, doc_id, request
            )

            # Deduplicate chunks
            await progress_tracker.start_step("deduplicating")
            dedup_start = datetime.now(timezone.utc)
            deduplicated_docs = await self._deduplicate_documents(
                internal_docs, request, progress_tracker
            )
            dedup_time = (datetime.now(timezone.utc) - dedup_start).total_seconds()
            duplicates_removed = len(internal_docs) - len(deduplicated_docs)
            await progress_tracker.complete_step(
                "deduplicating",
                f"Checked {len(internal_docs)} chunks, removed {duplicates_removed} duplicates in {dedup_time:.2f}s"
            )

            if not deduplicated_docs:
                raise ParsingError("All chunks were duplicates")
            
            # Separate table documents from regular documents
            table_docs = []
            regular_docs = []
            
            for doc in deduplicated_docs:
                # Get metadata dict properly
                if hasattr(doc.metadata, 'model_dump'):
                    metadata_dict = doc.metadata.model_dump()
                elif isinstance(doc.metadata, dict):
                    metadata_dict = doc.metadata
                else:
                    metadata_dict = {}
                
                content_type = metadata_dict.get("content_type", "")
                if content_type in ["table_markdown", "table_key_value", "table_html", "table_json", "table_unstructured"]:
                    table_docs.append(doc)
                else:
                    regular_docs.append(doc)
            
            logger.info(f"Separated documents: {len(table_docs)} tables, {len(regular_docs)} regular documents")
            
            # Add tables using multi-vector approach if any
            if table_docs and self.table_retriever:
                logger.info("Adding table documents with multi-vector approach")
                # Convert internal documents to LangChain documents for table retriever
                langchain_table_docs = []
                for doc in table_docs:
                    langchain_doc = LangchainDocument(
                        page_content=doc.content,
                        metadata=doc.metadata.model_dump() if hasattr(doc.metadata, 'model_dump') else doc.metadata
                    )
                    langchain_table_docs.append(langchain_doc)
                
                # Add tables to multi-vector retriever
                try:
                    await self.table_retriever.add_tables(langchain_table_docs)
                    logger.info(f"Added {len(langchain_table_docs)} tables to multi-vector retriever")
                except Exception as e:
                    # Table fallback alerting - log with structured data for monitoring
                    doc_source = request.url or request.file_path or "unknown"
                    logger.warning(
                        f"TABLE_FALLBACK: Failed to add {len(langchain_table_docs)} tables to multi-vector retriever. "
                        f"Source: {doc_source}. Error: {e}"
                    )
                    # Record metrics for monitoring
                    if self.performance_monitor:
                        self.performance_monitor.increment_counter("table_fallback_count", 1)
                        self.performance_monitor.record_event("table_fallback", {
                            "error": str(e),
                            "error_type": type(e).__name__,
                            "table_count": len(langchain_table_docs),
                            "doc_source": doc_source,
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        })
                    # Fall back to adding as regular documents (loses structured table benefits)
                    regular_docs.extend(table_docs)
                    logger.info(f"Fallback: {len(table_docs)} tables added as regular documents")
            elif table_docs:
                # No table retriever available, add as regular documents
                logger.info("Table retriever not available, adding tables as regular documents")
                regular_docs.extend(table_docs)
            
            # Add regular documents to vector store with parallel processing
            if regular_docs and (not checkpoint or checkpoint.current_state in [CheckpointState.LOADING, CheckpointState.SPLITTING, CheckpointState.EMBEDDING, CheckpointState.STORING]):
                # Filter out already processed chunks if resuming
                if checkpoint and checkpoint.processed_chunks:
                    regular_docs = [doc for doc in regular_docs if doc.id not in checkpoint.processed_chunks]
                    logger.info(f"Resuming with {len(regular_docs)} unprocessed documents")
                
                if regular_docs:
                    logger.info(f"Adding {len(regular_docs)} regular chunks to vector store with parallel processing")
                    await progress_tracker.start_step("embedding")
                    if self.checkpoint_service:
                        await self.checkpoint_service.update_progress(operation_id, new_state=CheckpointState.EMBEDDING)
                    await progress_tracker.start_step("storing")
                    if self.checkpoint_service:
                        await self.checkpoint_service.update_progress(operation_id, new_state=CheckpointState.STORING)
                    store_start = datetime.now(timezone.utc)
                    await self._store_documents_parallel(regular_docs, progress_tracker, checkpoint)
                    store_time = (datetime.now(timezone.utc) - store_start).total_seconds()
                    await progress_tracker.complete_step("storing", f"Stored {len(regular_docs)} documents in {store_time:.2f}s")
                    logger.info(f"Vector store addition completed in {store_time:.2f} seconds")

            # Refresh BM25 corpus cache so new documents are visible to BM25 retriever
            try:
                # This ensures subsequent retrieval pipelines see the updated corpus
                self.vector_store_manager.get_all_documents(refresh=True)
                logger.info("BM25 corpus cache refreshed after ingestion")
            except Exception as e:
                logger.warning(f"Failed to refresh BM25 corpus cache: {e}")

            # Update BM25 index with new documents
            # DISABLED: Incremental BM25 updates load entire corpus into memory (~3-4GB)
            # causing OOM kills during bulk ingestion. BM25 index should be rebuilt
            # once after all files are ingested instead.
            # await self._update_bm25_index(deduplicated_docs)
            
            # Update co-occurrence index with new documents
            await self._update_cooccurrence_index(deduplicated_docs)

            # Persist source catalog metadata for downstream retrieval
            await self._update_source_catalog(deduplicated_docs)
            
            # Cache document info
            if self.cache_service:
                await self._cache_document_info(doc_id, deduplicated_docs, request)
                
            # Calculate processing time
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            # Mark complete
            await progress_tracker.complete()
            
            # Mark checkpoint as completed
            if self.checkpoint_service:
                await self.checkpoint_service.mark_completed(operation_id)
            
            logger.info(
                f"Successfully ingested document {doc_id} with {len(internal_docs)} chunks "
                f"in {processing_time:.2f} seconds"
            )

            self.performance_monitor.record_ingestion_result(
                processing_time_ms=processing_time * 1000,
                chunks=len(deduplicated_docs),
                invalid_chunks=invalid_chunk_count,
                status="success",
            )

            return DocumentIngestionResponse(
                document_id=doc_id,
                chunks_created=len(deduplicated_docs),
                status="success",
                message=f"Successfully ingested document into {len(deduplicated_docs)} chunks",
                processing_time=processing_time,
                error_details={
                    "original_chunks": len(internal_docs),
                    "deduplicated_chunks": len(deduplicated_docs),
                    "duplicates_removed": len(internal_docs) - len(deduplicated_docs)
                } if len(internal_docs) != len(deduplicated_docs) else None
            )
            
        except IngestionError as e:
            # Already categorized error
            logger.error(f"Ingestion failed: {e.to_dict()}")
            if 'progress_tracker' in locals():
                await progress_tracker.error_step(
                    progress_tracker.current_step_id or "unknown",
                    str(e)
                )
            # Mark checkpoint as failed
            if self.checkpoint_service and 'operation_id' in locals():
                await self.checkpoint_service.mark_failed(operation_id, str(e))
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()

            self.performance_monitor.record_ingestion_result(
                processing_time_ms=processing_time * 1000,
                chunks=len(deduplicated_docs),
                invalid_chunks=invalid_chunk_count,
                status="error",
            )

            return DocumentIngestionResponse(
                document_id="",
                chunks_created=0,
                status="error",
                message=e.message,
                processing_time=processing_time,
                error_details=e.to_dict()
            )
            
        except Exception as e:
            # Categorize unknown errors
            categorized_error = categorize_error(e)
            logger.error(f"Ingestion failed: {categorized_error.to_dict()}")
            # Mark checkpoint as failed
            if self.checkpoint_service and 'operation_id' in locals():
                await self.checkpoint_service.mark_failed(operation_id, str(e))
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()

            self.performance_monitor.record_ingestion_result(
                processing_time_ms=processing_time * 1000,
                chunks=len(deduplicated_docs),
                invalid_chunks=invalid_chunk_count,
                status="error",
            )

            return DocumentIngestionResponse(
                document_id="",
                chunks_created=0,
                status="error",
                message=categorized_error.message,
                processing_time=processing_time,
                error_details=categorized_error.to_dict()
            )

        finally:
            self.performance_monitor.adjust_ingestion_in_progress(-1)
            
    async def ingest_batch(
        self,
        requests: List[DocumentIngestionRequest],
        max_concurrent: int = 5,
        progress_callback: Optional[callable] = None
    ) -> List[DocumentIngestionResponse]:
        """Ingest multiple documents concurrently."""
        if not requests:
            return []
            
        # Validate all requests first
        for i, request in enumerate(requests):
            try:
                self._validate_request(request)
            except ValidationError as e:
                # Return early with validation errors
                return [
                    DocumentIngestionResponse(
                        document_id="",
                        chunks_created=0,
                        status="error",
                        message=f"Request {i}: {e.message}",
                        processing_time=0,
                        error_details=e.to_dict()
                    )
                    for _ in requests
                ]
                
        # Create semaphore to limit concurrent operations
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def ingest_with_semaphore(request: DocumentIngestionRequest, index: int):
            """Ingest document with concurrency control."""
            async with semaphore:
                try:
                    response = await self.ingest_document(request)
                    
                    # Call progress callback if provided
                    if progress_callback:
                        await progress_callback(index, len(requests), response)
                        
                    return response
                    
                except Exception as e:
                    logger.error(f"Batch ingestion failed for request {index}: {e}")
                    # Return error response
                    return DocumentIngestionResponse(
                        document_id="",
                        chunks_created=0,
                        status="error",
                        message=str(e),
                        processing_time=0
                    )
                    
        # Create tasks for all requests
        tasks = [
            ingest_with_semaphore(request, i) 
            for i, request in enumerate(requests)
        ]
        
        # Execute all tasks concurrently
        responses = await asyncio.gather(*tasks, return_exceptions=False)
        
        # Log batch statistics
        successful = sum(1 for r in responses if r.status == "success")
        failed = sum(1 for r in responses if r.status == "error")
        existing = sum(1 for r in responses if r.status == "exists")
        
        logger.info(
            f"Batch ingestion complete: {successful} successful, "
            f"{failed} failed, {existing} already existing out of {len(requests)} total"
        )
        
        return responses
        
    async def ingest_canada_ca(self) -> DocumentIngestionResponse:
        """Ingest all Canada.ca travel instructions."""
        start_time = datetime.now(timezone.utc)
        
        try:
            # Use specialized scraper
            scraper = CanadaCaScraper()
            documents = await scraper.scrape_travel_instructions()
            
            # Split all documents using LangChain splitter
            splitter = LangChainTextSplitter()
            all_chunks = splitter.split_documents(documents)
            
            # Generate parent document ID
            doc_id = f"canada_ca_travel_{datetime.now(timezone.utc).strftime('%Y%m%d')}"
            
            # Convert to internal format
            internal_docs = []
            for i, chunk in enumerate(all_chunks):
                metadata_dict = dict(chunk.metadata or {})
                source_fields = self._prepare_source_metadata(metadata_dict, None)
                tags_value = metadata_dict.get("tags") or ["canada.ca", "travel", "policy", "official"]
                if isinstance(tags_value, str):
                    tags_value = [tags_value]

                metadata = DocumentMetadata(
                    source=source_fields.get("source", metadata_dict.get("source", "")),
                    title=metadata_dict.get("title", "Canadian Forces Travel Instructions"),
                    type=DocumentType.WEB,
                    section=metadata_dict.get("section"),
                    last_modified=metadata_dict.get("last_modified"),
                    policy_reference=metadata_dict.get("policy_reference"),
                    tags=tags_value,
                    source_id=source_fields.get("source_id"),
                    canonical_url=source_fields.get("canonical_url"),
                    reference_path=source_fields.get("reference_path"),
                )
                
                internal_doc = Document(
                    id=f"{doc_id}_chunk_{i}",
                    content=chunk.page_content,
                    metadata=metadata,
                    chunk_index=i,
                    parent_id=doc_id,
                    created_at=datetime.now(timezone.utc)
                )
                internal_docs.append(internal_doc)
                
            # Add to vector store
            logger.info(f"Adding {len(internal_docs)} Canada.ca chunks to vector store")
            await self.vector_store_manager.add_documents(internal_docs)

            # Update source catalog so Canada.ca content participates in citations
            await self._update_source_catalog(internal_docs)
            
            # Cache ingestion info
            if self.cache_service:
                await self.cache_service.set(
                    f"doc:{doc_id}",
                    {
                        "id": doc_id,
                        "chunks": len(internal_docs),
                        "source": "canada.ca",
                        "pages_scraped": len(documents),
                        "ingested_at": datetime.now(timezone.utc).isoformat()
                    },
                    ttl=86400
                )
                
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            return DocumentIngestionResponse(
                document_id=doc_id,
                chunks_created=len(internal_docs),
                status="success",
                message=f"Successfully ingested {len(documents)} Canada.ca pages into {len(internal_docs)} chunks",
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Canada.ca ingestion failed: {e}")
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            return DocumentIngestionResponse(
                document_id="",
                chunks_created=0,
                status="error",
                message=str(e),
                processing_time=processing_time
            )

    def _extract_column_number(self, chunk: Document) -> Optional[Dict[str, Any]]:
        """Extract column number metadata from DOA document chunks.

        Args:
            chunk: Document chunk to extract from

        Returns:
            Dict with column_number and column_name if found, else None
        """
        import re

        text = chunk.page_content
        if not text:
            return None

        # Pattern to match "Column X –" or "Column X:" format
        # The DOA PDF uses "Column 17 – Services (Competitive) – General:"
        column_pattern = r'Column\s+(\d+)\s+[–:-]\s+([^\n]+)'

        match = re.search(column_pattern, text)
        if match:
            column_num = match.group(1)
            column_name = match.group(2).strip()

            # Clean up column name (remove trailing punctuation)
            column_name = re.sub(r'[:\s]+$', '', column_name)

            return {
                "column_number": column_num,
                "column_name": column_name
            }

        return None

    def _add_column_context_to_chunks(self, chunks: List[Document], request: DocumentIngestionRequest) -> List[Document]:
        """Add column headers to DOA table chunks that lost them during PDF extraction.

        Args:
            chunks: List of document chunks
            request: Ingestion request with file path info

        Returns:
            List of chunks with column context added where applicable
        """
        import re

        # Only process delegation of authority documents
        source = request.metadata.get('source', '') if request.metadata else ''
        if 'delegation' not in source.lower():
            return chunks

        # Column patterns: keywords that indicate which column a chunk belongs to
        column_patterns = {
            17: {
                "keywords": ["Services (Competitive)", "competitive) – general", "competitive general"],
                "anti_keywords": ["non-competitive"],  # Don't match if this is present
                "header": "Column 17 – Services (Competitive) – General:\n\n"
            },
            18: {
                "keywords": ["Services (Non-Competitive)", "non-competitive) – general", "non-competitive general"],
                "anti_keywords": [],
                "header": "Column 18 – Services (Non-Competitive) – General:\n\n"
            }
        }

        modified_count = 0
        for chunk in chunks:
            text = chunk.page_content
            text_lower = text.lower()

            # Skip if already has column header
            if re.match(r'^Column\s+\d+\s+[–:-]', text):
                continue

            # Check each column pattern
            for col_num, pattern_info in column_patterns.items():
                # Check if any keyword matches
                has_keyword = any(kw.lower() in text_lower for kw in pattern_info['keywords'])

                # Check if any anti-keyword matches (exclude if present)
                has_anti_keyword = any(akw.lower() in text_lower for akw in pattern_info['anti_keywords'])

                if has_keyword and not has_anti_keyword:
                    # Prepend column header
                    chunk.page_content = pattern_info['header'] + text

                    # Update metadata
                    chunk.metadata['column_number'] = str(col_num)
                    chunk.metadata['column_context_added'] = True

                    modified_count += 1
                    break

        if modified_count > 0:
            logger.info(f"Added column context to {modified_count} chunks")

        return chunks

    def _generate_document_id(self, request: DocumentIngestionRequest) -> str:
        """Generate unique document ID using content-based hashing."""
        # For content-based ID, use the content itself
        if request.content:
            # Use content hash for direct input
            content_hash = self.content_hasher.generate_content_hash(request.content)
            return f"doc_{content_hash[:12]}"
            
        # For URL/file, combine source and metadata
        if request.url:
            source = request.url
        elif request.file_path:
            source = request.file_path
        else:
            source = str(uuid.uuid4())
            
        # Include metadata in hash for better uniqueness
        hash_input = f"{source}:{request.type}:{str(request.metadata)}"
        hash_obj = hashlib.sha256(hash_input.encode())
        return f"doc_{hash_obj.hexdigest()[:12]}"
        
    async def _check_existing_document(
        self,
        request: DocumentIngestionRequest
    ) -> Optional[Dict[str, Any]]:
        """Check if document already exists."""
        if not self.cache_service:
            return None
            
        doc_id = self._generate_document_id(request)
        cached = await self.cache_service.get(f"doc:{doc_id}")
        
        return cached
        
    def _validate_request(self, request: DocumentIngestionRequest) -> None:
        """Validate ingestion request."""
        # Must have either URL, file path, or content
        if not request.url and not request.file_path and not request.content:
            raise ValidationError(
                "Must provide either URL, file path, or content",
                field="source"
            )
            
        # Validate URL format if provided
        if request.url:
            import re
            url_pattern = re.compile(
                r'^https?://'  # http:// or https://
                r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
                r'localhost|'  # localhost...
                r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
                r'(?::\d+)?'  # optional port
                r'(?:/?|[/?]\S+)$', re.IGNORECASE
            )
            if not url_pattern.match(request.url):
                raise ValidationError(
                    f"Invalid URL format: {request.url}",
                    field="url",
                    value=request.url
                )
                
        # Validate file path if provided
        if request.file_path:
            import os
            if not os.path.exists(request.file_path):
                raise ValidationError(
                    f"File not found: {request.file_path}",
                    field="file_path",
                    value=request.file_path
                )
                
        # Validate document type
        if request.type not in DocumentType:
            raise ValidationError(
                f"Invalid document type: {request.type}",
                field="type",
                value=request.type
            )
            
    async def _load_documents_with_retry(
        self, 
        request: DocumentIngestionRequest
    ) -> List[LangchainDocument]:
        """Load documents with retry logic."""
        async def load_documents():
            if request.content:
                # Direct content input
                return [LangchainDocument(
                    page_content=request.content,
                    metadata={
                        "source": "direct_input",
                        "type": request.type,
                        **request.metadata
                    }
                )]
            else:
                # Load from URL or file
                loader = LangChainDocumentLoader()
                if request.url:
                    return await loader.load_from_url(request.url)
                else:
                    return await loader.load_from_file(request.file_path)
                
        # Use retry manager for network operations
        return await self.retry_manager.execute_with_retry_async(
            load_documents
        )
        
    async def _split_documents_safely(
        self,
        documents: List[LangchainDocument],
        progress_tracker: Optional[IngestionProgressTracker] = None
    ) -> List[LangchainDocument]:
        """Split documents with error handling, smart chunking, and parallel processing."""
        try:
            # Temporary debugging
            logger.error(f"DEBUG: _split_documents_safely called with {len(documents)} documents")
            for i, doc in enumerate(documents):
                logger.error(f"DEBUG: Document {i}: type={type(doc)}, has_page_content={hasattr(doc, 'page_content')}")
                if hasattr(doc, 'page_content'):
                    logger.error(f"DEBUG: Document {i}: page_content_length={len(doc.page_content)}")
                else:
                    logger.error(f"DEBUG: Document {i}: MISSING page_content! doc={doc}")
            # Initialize the appropriate splitter
            if self.use_smart_chunking:
                # Use smart document splitter for structured documents
                splitter = SmartDocumentSplitter()
            else:
                # Use standard LangChain splitter
                splitter = LangChainTextSplitter()
            
            # For large documents, split in parallel
            if len(documents) > 5 or sum(len(doc.page_content) for doc in documents) > 50000:
                logger.info(f"Using parallel processing to split {len(documents)} documents")
                
                # Define splitting function
                def split_single_doc(doc: LangchainDocument) -> List[LangchainDocument]:
                    # Check if document type supports type-aware splitting
                    doc_type = doc.metadata.get("type", DocumentType.TEXT)
                    
                    # Use the appropriate splitting method
                    if isinstance(splitter, SmartDocumentSplitter):
                        # SmartDocumentSplitter.split_by_type expects a single Document, not a list
                        return splitter.split_by_type(doc, doc_type)
                    else:
                        # LangChainTextSplitter has split_documents method which expects a list
                        return splitter.split_documents([doc])
                
                # Process documents in parallel using executor
                loop = asyncio.get_event_loop()
                tasks = []
                for doc in documents:
                    task = loop.run_in_executor(
                        self.chunk_processor.executor,
                        split_single_doc,
                        doc
                    )
                    tasks.append(task)
                
                # Wait for all documents to be split with progress updates
                completed = 0
                chunks = []
                for future in asyncio.as_completed(tasks):
                    doc_chunks = await future
                    chunks.extend(doc_chunks)
                    completed += 1
                    
                    # Update progress
                    if progress_tracker:
                        await progress_tracker.update_splitting_progress(
                            completed,
                            len(documents)
                        )
                
                logger.info(f"Parallel splitting produced {len(chunks)} chunks")
                return chunks
            
            # For smaller documents, use standard processing
            else:
                # Check document type
                if documents:
                    doc_type = documents[0].metadata.get("type", DocumentType.TEXT)
                    
                    # Use the appropriate splitting method
                    if isinstance(splitter, SmartDocumentSplitter):
                        # Process each document individually for SmartDocumentSplitter
                        logger.info(f"Using SmartDocumentSplitter with {len(documents)} documents")
                        chunks = []
                        for i, doc in enumerate(documents):
                            logger.info(f"Processing document {i}: type={type(doc)}, has_page_content={hasattr(doc, 'page_content')}")
                            if hasattr(doc, 'page_content'):
                                logger.info(f"Document {i} content length: {len(doc.page_content) if doc.page_content else 0}")
                            else:
                                logger.error(f"Document {i} is missing page_content attribute: {doc}")
                            doc_chunks = splitter.split_by_type(doc, doc_type)
                            chunks.extend(doc_chunks)
                        return chunks
                    else:
                        # LangChainTextSplitter can handle a list
                        return splitter.split_documents(documents)
                else:
                    return []
            
        except Exception as e:
            raise ParsingError(
                f"Failed to split documents: {e}",
                document_type="multiple"
            )
            
    def _convert_to_internal_documents(
        self,
        chunks: List[LangchainDocument],
        doc_id: str,
        request: DocumentIngestionRequest
    ) -> List[Document]:
        """Convert LangChain documents to internal format."""
        internal_docs = []
        
        for i, chunk in enumerate(chunks):
            try:
                source_fields = self._prepare_source_metadata(
                    chunk.metadata,
                    request
                )

                # Extract CLI-provided metadata (from ingest_sources_cli.py)
                source_identity = chunk.metadata.get("source_identity")
                document_info = chunk.metadata.get("document_info")
                structure_info = chunk.metadata.get("structure_info")

                # Enhance metadata - use CLI extractions when available
                metadata = DocumentMetadata(
                    source=source_fields.get("source", "direct_input"),
                    title=(
                        document_info.get("title") if document_info and document_info.get("title")
                        else chunk.metadata.get("title")
                    ),
                    type=DocumentType(chunk.metadata.get("type", request.type)),
                    section=chunk.metadata.get("section"),
                    page_number=chunk.metadata.get("page"),
                    last_modified=chunk.metadata.get("last_modified"),
                    policy_reference=chunk.metadata.get("policy_reference"),
                    tags=chunk.metadata.get("tags", []),
                    source_id=source_fields.get("source_id"),
                    canonical_url=(
                        document_info.get("canonical_url") if document_info and document_info.get("canonical_url")
                        else source_fields.get("canonical_url")
                    ),
                    reference_path=source_fields.get("reference_path"),
                    # Preserve CLI extractions for better citations
                    source_identity=source_identity,
                    document_info=document_info,
                    publisher_confidence=source_identity.get("confidence") if source_identity else None,
                    evidence=source_identity.get("evidence") if source_identity else None,
                    structure_info=structure_info,
                )
                
                # Create internal document
                internal_doc = Document(
                    id=f"{doc_id}_chunk_{i}",
                    content=chunk.page_content,
                    metadata=metadata,
                    chunk_index=i,
                    parent_id=doc_id,
                    created_at=datetime.now(timezone.utc)
                )
                internal_docs.append(internal_doc)
                
            except Exception as e:
                logger.warning(f"Failed to convert chunk {i}: {e}")
                # Continue with other chunks
                
        if not internal_docs:
            raise ParsingError("Failed to convert any chunks to internal format")
            
        return internal_docs
        
    def _prepare_source_metadata(
        self,
        raw_metadata: Dict[str, Any],
        request: Optional[DocumentIngestionRequest] = None
    ) -> Dict[str, Optional[str]]:
        """Derive stable source metadata for cataloging and citations."""

        metadata = raw_metadata or {}

        source_value = metadata.get("source")
        if not source_value and request:
            source_value = request.url or request.file_path or "direct_input"
        if not source_value:
            source_value = "direct_input"
        canonical_url = metadata.get("canonical_url")

        if not canonical_url and isinstance(source_value, str) and source_value.startswith("http"):
            canonical_url = source_value

        reference_path = metadata.get("reference_path") or metadata.get("section") or metadata.get("title")

        source_identifier = metadata.get("source_id")
        if not source_identifier:
            identifier_parts = [
                str(part)
                for part in [
                    canonical_url or source_value,
                    reference_path,
                    metadata.get("policy_reference"),
                    metadata.get("document_id"),
                    (request.type if request else metadata.get("type")),
                ]
                if part
            ]
            identifier_seed = "|".join(identifier_parts).strip()

            if not identifier_seed:
                inferred_type = request.type if request else metadata.get("type") or "unknown"
                identifier_seed = f"{inferred_type}:{source_value}"

            source_identifier = hashlib.sha1(identifier_seed.encode("utf-8")).hexdigest()

        return {
            "source": str(source_value) if source_value is not None else "direct_input",
            "canonical_url": str(canonical_url) if canonical_url else None,
            "reference_path": str(reference_path) if reference_path else None,
            "source_id": source_identifier,
        }

    def _serialize_documents_for_checkpoint(
        self,
        documents: List[LangchainDocument]
    ) -> List[Dict[str, Any]]:
        """Convert LangChain documents into JSON-serializable dictionaries."""
        serialized = []
        max_len = getattr(settings, "checkpoint_content_max_chars", 4000)
        for doc in documents:
            content = doc.page_content or ""
            if max_len and len(content) > max_len:
                content = content[:max_len]
            serialized.append(
                {
                    "page_content": content,
                    "metadata": doc.metadata or {},
                }
            )
        return serialized

    def _deserialize_documents_from_checkpoint(
        self,
        serialized: List[Dict[str, Any]]
    ) -> List[LangchainDocument]:
        """Rehydrate serialized documents into LangChain Document instances."""
        documents = []
        for item in serialized or []:
            documents.append(
                LangchainDocument(
                    page_content=item.get("page_content", ""),
                    metadata=item.get("metadata", {})
                )
            )
        return documents

    def _serialize_chunks_for_checkpoint(
        self,
        chunks: List[LangchainDocument]
    ) -> List[Dict[str, Any]]:
        """Serialize chunk objects for persistence."""
        return self._serialize_documents_for_checkpoint(chunks)

    def _deserialize_chunks_from_checkpoint(
        self,
        serialized: List[Dict[str, Any]]
    ) -> List[LangchainDocument]:
        """Deserialize chunk payload stored in checkpoint metadata."""
        return self._deserialize_documents_from_checkpoint(serialized)

    async def _store_documents_with_retry(
        self, 
        documents: List[Document]
    ) -> None:
        """Store documents in vector store with retry."""
        async def store_documents():
            try:
                await self.vector_store_manager.add_documents(documents)
            except Exception as e:
                # Convert to storage error for proper categorization
                raise StorageError(
                    f"Failed to store documents: {e}",
                    operation="add_documents"
                )
                
        await self.retry_manager.execute_with_retry_async(
            store_documents
        )
        
    async def _store_documents_parallel(
        self, 
        documents: List[Document],
        progress_tracker: Optional[IngestionProgressTracker] = None,
        checkpoint: Optional[Any] = None
    ) -> None:
        """Store documents with parallel embedding generation."""
        try:
            # Initialize optimized writer if not already done
            if not self.optimized_writer:
                self.optimized_writer = OptimizedVectorStoreWriter(
                    self.vector_store_manager.vector_store,
                    self.vector_store_manager.embeddings,
                    progress_tracker=progress_tracker,
                    cache_service=self.embedding_cache
                )
            
            # Set checkpoint callback if available
            if checkpoint and self.checkpoint_service:
                async def checkpoint_callback(doc_id: str):
                    await self.checkpoint_service.update_progress(
                        checkpoint.operation_id,
                        processed_chunk_id=doc_id
                    )
                self.optimized_writer.set_checkpoint_callback(checkpoint_callback)
            
            # Use optimized parallel processing with configuration values
            await self.optimized_writer.add_documents_optimized(
                documents,
                batch_size=settings.vector_store_batch_size,
                embedding_batch_size=settings.embedding_batch_size,
                max_concurrent_embeddings=settings.max_concurrent_embeddings
            )
            
        except Exception as e:
            logger.warning(f"Parallel storage failed, falling back to standard storage: {e}")
            # Fall back to standard storage with retry
            await self._store_documents_with_retry(documents)
        
    async def _cache_document_info(
        self,
        doc_id: str,
        internal_docs: List[Document],
        request: DocumentIngestionRequest
    ) -> None:
        """Cache document information."""
        try:
            await self.cache_service.set(
                f"doc:{doc_id}",
                {
                    "id": doc_id,
                    "chunks": len(internal_docs),
                    "source": request.url or request.file_path or "direct",
                    "type": request.type,
                    "ingested_at": datetime.now(timezone.utc).isoformat(),
                    "metadata": request.metadata
                },
                ttl=86400  # 24 hours
            )
        except Exception as e:
            # Log but don't fail ingestion for cache errors
            logger.warning(f"Failed to cache document info: {e}")

    async def _update_source_catalog(self, documents: List[Document]) -> None:
        """Sync ingested documents to the canonical source repository."""
        if not self.source_repository or not documents:
            return

        try:
            entries: Dict[str, SourceCatalogEntry] = {}

            def _extract_table_metadata(document: Document) -> Optional[Dict[str, Any]]:
                metadata = document.metadata.model_dump() if hasattr(document.metadata, 'model_dump') else document.metadata
                if isinstance(metadata, dict):
                    content_type = (metadata.get("content_type", "") or "").lower()
                else:
                    content_type = ""
                content = document.content or ""
                if "table" not in content_type and "|" not in content:
                    return None
                lines = [line for line in content.splitlines() if line.strip()]
                table_lines = [line for line in lines if "|" in line]
                if not table_lines:
                    return None
                header_line = table_lines[0]
                headers = [col.strip() for col in header_line.strip("|").split("|") if col.strip()]
                sample_rows = table_lines[: min(len(table_lines), 5)]
                return {
                    "headers": headers,
                    "sample": "\n".join(sample_rows),
                }

            for doc in documents:
                metadata_dict = doc.metadata.model_dump() if hasattr(doc.metadata, 'model_dump') else dict(doc.metadata)
                source_id = metadata_dict.get("source_id")

                if not source_id:
                    continue

                entry = entries.get(source_id)
                if not entry:
                    entry = SourceCatalogEntry(
                        source_id=source_id,
                        title=metadata_dict.get("title"),
                        canonical_url=metadata_dict.get("canonical_url") or metadata_dict.get("source"),
                        reference_path=metadata_dict.get("reference_path") or metadata_dict.get("section"),
                        document_type=str(metadata_dict.get("type")) if metadata_dict.get("type") else None,
                        section=metadata_dict.get("section"),
                        metadata={
                            "source": metadata_dict.get("source"),
                            "policy_reference": metadata_dict.get("policy_reference"),
                            "page_number": metadata_dict.get("page_number"),
                            "tags": metadata_dict.get("tags"),
                        },
                    )
                    entries[source_id] = entry

                entry.last_ingested_at = datetime.now(timezone.utc)
                parent_ref = doc.parent_id or metadata_dict.get("parent_id")
                entry.register_chunk(str(doc.id), str(parent_ref) if parent_ref else None)

                table_meta = _extract_table_metadata(doc)
                if table_meta:
                    entry.metadata.setdefault("has_table", True)
                    tables = entry.metadata.setdefault("tables", [])
                    if len(tables) < 5:
                        tables.append(table_meta)

            if entries:
                await self.source_repository.upsert_entries(entries.values())

        except Exception as exc:
            logger.warning("Failed to update source repository: %s", exc)

    async def _deduplicate_documents(
        self,
        documents: List[Document],
        request: DocumentIngestionRequest,
        progress_tracker=None
    ) -> List[Document]:
        """Deduplicate documents against existing content."""
        if not documents:
            return documents

        # Check if we should skip deduplication
        if request.force_refresh:
            logger.info("Skipping deduplication due to force_refresh=True")
            if progress_tracker:
                await progress_tracker.update_deduplication_progress(len(documents), len(documents), 0)
            return documents

        try:
            # Convert documents to format expected by deduplication service
            docs_for_dedup = []
            for doc in documents:
                docs_for_dedup.append({
                    "id": doc.id,
                    "content": doc.content,
                    "metadata": doc.metadata.model_dump() if hasattr(doc.metadata, 'model_dump') else doc.metadata
                })

            # Check against existing documents in vector store
            existing_docs = await self._get_existing_documents_for_dedup(
                request, limit=100
            )

            if existing_docs:
                # Check each new document against existing ones
                duplicates_to_remove = set()
                total_docs = len(docs_for_dedup)

                for idx, new_doc in enumerate(docs_for_dedup):
                    # Report progress every chunk
                    if progress_tracker and idx % 10 == 0:  # Update every 10 chunks to avoid too many updates
                        await progress_tracker.update_deduplication_progress(
                            idx, total_docs, len(duplicates_to_remove)
                        )

                    for existing_doc in existing_docs:
                        is_dup, score, reason = self.deduplication_service.is_duplicate(
                            new_doc["content"],
                            existing_doc.get("content", ""),
                            new_doc.get("metadata"),
                            existing_doc.get("metadata")
                        )

                        if is_dup and reason != "updated_version":
                            logger.info(
                                f"Found duplicate chunk {new_doc['id']}: "
                                f"score={score:.2f}, reason={reason}"
                            )
                            duplicates_to_remove.add(new_doc["id"])
                            break

                # Final progress update
                if progress_tracker:
                    await progress_tracker.update_deduplication_progress(
                        total_docs, total_docs, len(duplicates_to_remove)
                    )

                # Remove duplicates
                if duplicates_to_remove:
                    documents = [
                        doc for doc in documents
                        if doc.id not in duplicates_to_remove
                    ]
                    logger.info(
                        f"Removed {len(duplicates_to_remove)} duplicate chunks"
                    )
            else:
                # No existing docs to check against
                if progress_tracker:
                    await progress_tracker.update_deduplication_progress(len(documents), len(documents), 0)
                    
            # Also deduplicate within the batch itself
            deduplicated = self.deduplication_service.deduplicate_chunks(
                docs_for_dedup, strategy="merge"
            )
            
            # Convert back to Document objects
            final_docs = []
            for dedup_doc in deduplicated:
                # Find original document
                original = next(
                    (doc for doc in documents if doc.id == dedup_doc["id"]),
                    None
                )
                if original:
                    # Update metadata if it was merged
                    if "metadata" in dedup_doc:
                        # Update the original document's metadata
                        for key, value in dedup_doc["metadata"].items():
                            if hasattr(original.metadata, key):
                                setattr(original.metadata, key, value)
                    final_docs.append(original)
                    
            return final_docs
            
        except Exception as e:
            logger.warning(f"Deduplication failed: {e}. Proceeding without deduplication.")
            return documents
            
    async def _get_existing_documents_for_dedup(
        self,
        request: DocumentIngestionRequest,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get existing documents for deduplication check."""
        try:
            lookup_values: List[str] = []
            if request.url:
                lookup_values.append(request.url)
            if request.file_path:
                lookup_values.append(request.file_path)
            if request.metadata:
                candidate = request.metadata.get("source") or request.metadata.get("canonical_url")
                if candidate:
                    lookup_values.append(candidate)

            if not lookup_values:
                return []

            lookup_values = list({str(value) for value in lookup_values if value})

            def _fetch_matching_docs() -> List[Dict[str, Any]]:
                all_docs = self.vector_store_manager.get_all_documents(refresh=False)
                matches: List[Dict[str, Any]] = []
                for doc in all_docs:
                    metadata = doc.metadata or {}
                    source_candidates = {
                        metadata.get("source"),
                        metadata.get("canonical_url"),
                        metadata.get("file_path"),
                    }
                    if any(value in lookup_values for value in source_candidates if value):
                        matches.append(
                            {
                                "id": metadata.get("id"),
                                "content": doc.page_content,
                                "metadata": metadata,
                            }
                        )
                        if len(matches) >= limit:
                            break
                return matches

            return await asyncio.to_thread(_fetch_matching_docs)

        except Exception as e:
            logger.warning(f"Failed to get existing documents: {e}")
            return []
            
    async def _update_bm25_index(self, documents: List[Document]) -> None:
        """Update BM25 index with new documents."""
        try:
            # Initialize BM25 retriever
            bm25_retriever = TravelBM25Retriever(documents=[])
            
            # Load existing index
            if bm25_retriever.load_index():
                logger.info("Loaded existing BM25 index for update")
                
                # Get all existing documents
                all_docs = list(bm25_retriever.documents)
                
                # Convert Document objects to LangchainDocument format
                new_langchain_docs = []
                for doc in documents:
                    langchain_doc = LangchainDocument(
                        page_content=doc.content,
                        metadata=doc.metadata.model_dump() if hasattr(doc.metadata, 'model_dump') else doc.metadata
                    )
                    new_langchain_docs.append(langchain_doc)
                
                # Add new documents
                all_docs.extend(new_langchain_docs)
                
                # Rebuild index with all documents
                bm25_retriever.build_index(all_docs)
                logger.info(f"Updated BM25 index with {len(documents)} new documents. Total documents: {len(all_docs)}")
            else:
                # No existing index, build new one with just these documents
                langchain_docs = []
                for doc in documents:
                    langchain_doc = LangchainDocument(
                        page_content=doc.content,
                        metadata=doc.metadata.model_dump() if hasattr(doc.metadata, 'model_dump') else doc.metadata
                    )
                    langchain_docs.append(langchain_doc)
                    
                bm25_retriever.build_index(langchain_docs)
                logger.info(f"Created new BM25 index with {len(documents)} documents")
                
        except Exception as e:
            logger.error(f"Failed to update BM25 index: {e}")
            # Don't fail the ingestion if BM25 update fails
            pass
            
    async def _update_cooccurrence_index(self, documents: List[Document]) -> None:
        """Update co-occurrence index with new documents."""
        try:
            # Convert Document objects to LangchainDocument format
            langchain_docs = []
            for doc in documents:
                # Prepare metadata
                if hasattr(doc.metadata, 'model_dump'):
                    metadata = doc.metadata.model_dump()
                elif isinstance(doc.metadata, dict):
                    metadata = doc.metadata
                else:
                    metadata = {}
                
                # Include document ID in metadata
                metadata['id'] = doc.id
                
                langchain_doc = LangchainDocument(
                    page_content=doc.content,
                    metadata=metadata
                )
                langchain_docs.append(langchain_doc)
            
            # Add documents to co-occurrence index
            for doc in langchain_docs:
                self.cooccurrence_indexer.add_document(doc)
            
            # Save the index
            self.cooccurrence_indexer.save_index()
            logger.info(f"Updated co-occurrence index with {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"Failed to update co-occurrence index: {e}")
            # Don't fail the ingestion if co-occurrence update fails
            pass
