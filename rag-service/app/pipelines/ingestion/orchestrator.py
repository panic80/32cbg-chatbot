"""Main ingestion pipeline orchestrator."""

import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from langchain_core.documents import Document as LangchainDocument

from app.core.config import settings
from app.core.errors import ParsingError, ValidationError
from app.core.logging import get_logger
from app.models.documents import (
    Document,
    DocumentType,
    DocumentIngestionRequest,
    DocumentIngestionResponse,
)
from app.models.source_catalog import SourceCatalogEntry
from app.pipelines.ingestion.document_loader import DocumentLoader
from app.pipelines.ingestion.chunker import DocumentChunker
from app.pipelines.ingestion.metadata_enricher import IngestionMetadataEnricher
from app.pipelines.ingestion.deduplicator import Deduplicator
from app.pipelines.ingestion.vector_writer import VectorWriter
from app.pipelines.ingestion.progress_manager import ProgressManager
from app.services.performance_monitor import get_performance_monitor
from app.services.quality_validator import ChunkQualityValidator
from app.services.ingestion_checkpoint import CheckpointState
from app.components.cooccurrence_indexer import CooccurrenceIndexer
from app.utils.deduplication import ContentHasher

if TYPE_CHECKING:
    from app.services.source_repository import SourceRepository

logger = get_logger(__name__)


class IngestionOrchestrator:
    """Orchestrates the document ingestion pipeline.

    Coordinates document loading, chunking, enrichment, deduplication,
    and storage while managing progress and checkpoints.
    """

    def __init__(
        self,
        vector_store_manager: Any,
        cache_service: Optional[Any] = None,
        source_repository: Optional["SourceRepository"] = None,
        llm: Optional[Any] = None,
        use_smart_chunking: bool = True,
        deduplication_threshold: float = 0.85,
    ):
        """Initialize ingestion orchestrator.

        Args:
            vector_store_manager: Vector store manager instance.
            cache_service: Optional cache service.
            source_repository: Optional source repository.
            llm: Optional LLM for metadata extraction.
            use_smart_chunking: Whether to use smart chunking.
            deduplication_threshold: Similarity threshold for dedup.
        """
        self._vector_store = vector_store_manager
        self._cache_service = cache_service
        self._source_repository = source_repository
        self._performance_monitor = get_performance_monitor()

        # Initialize components
        self._loader = DocumentLoader()
        self._chunker = DocumentChunker(use_smart_chunking=use_smart_chunking)
        self._enricher = IngestionMetadataEnricher(llm=llm)
        self._deduplicator = Deduplicator(
            threshold=deduplication_threshold,
            vector_store_manager=vector_store_manager,
        )

        # Initialize embedding cache
        embedding_cache = None
        if cache_service and getattr(settings, "enable_embedding_cache", True):
            from app.services.embedding_cache import EmbeddingCacheService

            embedding_cache = EmbeddingCacheService(
                cache_service, ttl=settings.embedding_cache_ttl
            )

        self._writer = VectorWriter(
            vector_store_manager,
            embedding_cache=embedding_cache,
        )
        self._progress = ProgressManager(cache_service)

        # Quality validator
        self._quality_validator = ChunkQualityValidator(
            min_chunk_size=settings.min_chunk_size,
            max_chunk_size=settings.max_chunk_size,
            min_quality_score=getattr(settings, "min_quality_score", 60.0),
        )

        # Co-occurrence indexer
        self._cooccurrence_indexer = CooccurrenceIndexer(
            index_path=Path("cooccurrence_index")
        )
        self._cooccurrence_indexer.load_index()

        # Table retriever (lazy init)
        self._table_retriever = None
        if llm and getattr(settings, "enable_table_multivector", True):
            try:
                from app.components.table_multi_vector_retriever import (
                    TableMultiVectorRetriever,
                )

                self._table_retriever = TableMultiVectorRetriever(
                    vectorstore=vector_store_manager.vector_store,
                    llm=llm,
                )
            except Exception as e:
                logger.warning(f"Failed to initialize table retriever: {e}")

        self._content_hasher = ContentHasher()

    async def cleanup(self):
        """Clean up resources."""
        self._chunker.close()
        self._writer.close()

    async def ingest(
        self,
        request: DocumentIngestionRequest,
        progress_callback: Optional[callable] = None,
    ) -> DocumentIngestionResponse:
        """Ingest a document through the pipeline.

        Args:
            request: The ingestion request.
            progress_callback: Optional progress callback.

        Returns:
            Ingestion response with results.
        """
        start_time = datetime.now(timezone.utc)
        self._performance_monitor.adjust_ingestion_in_progress(+1)
        self._performance_monitor.increment_counter("ingestion_started", 1)

        operation_id = f"ingest_{int(start_time.timestamp())}"
        tracker = self._progress.create_tracker(
            operation_id,
            request.url or request.file_path or "direct_input",
            progress_callback,
        )

        invalid_chunk_count = 0
        deduplicated_docs: List[Document] = []

        try:
            # Validate request
            self._validate_request(request)

            # Check for resumable checkpoint
            checkpoint = await self._progress.check_resumable(
                getattr(request, "operation_id", None) or ""
            )

            if not checkpoint:
                checkpoint = await self._progress.create_checkpoint(
                    operation_id,
                    request.url or request.file_path or "direct_input",
                )

            # Check if document exists
            if not request.force_refresh and not checkpoint:
                existing = await self._check_existing(request)
                if existing:
                    return self._create_exists_response(existing, start_time)

            # Load documents
            await tracker.start_step("loading")
            await self._progress.update_state(operation_id, CheckpointState.LOADING)

            documents = await self._loader.load(request)

            await tracker.complete_step("loading", f"Loaded {len(documents)} document(s)")

            if not documents:
                raise ParsingError("No content extracted from document")

            # Chunk documents
            await tracker.start_step("splitting")
            await self._progress.update_state(operation_id, CheckpointState.SPLITTING)

            chunks = await self._chunker.chunk_documents(
                documents,
                progress_callback=lambda done, total: tracker.update_splitting_progress(
                    done, total
                ),
            )

            await tracker.complete_step("splitting", f"Created {len(chunks)} chunks")

            if not chunks:
                raise ParsingError("No chunks created from document")

            # Enrich metadata
            chunks = await self._enricher.enrich_chunks(chunks, request)

            # Validate quality
            if getattr(settings, "enable_quality_validation", True):
                valid_chunks, invalid_chunks, stats = self._quality_validator.validate_batch(
                    chunks
                )
                invalid_chunk_count = len(invalid_chunks)
                logger.info(
                    f"Quality validation: {stats['valid_chunks']} valid, "
                    f"{stats['invalid_chunks']} invalid"
                )

                if getattr(settings, "strict_quality_validation", False):
                    chunks = valid_chunks
                    if not chunks:
                        raise ParsingError("No chunks passed quality validation")

            # Generate document ID and convert
            doc_id = self._generate_document_id(request)
            internal_docs = self._enricher.convert_to_internal_documents(
                chunks, doc_id, request
            )

            # Deduplicate
            await tracker.start_step("deduplicating")
            deduplicated_docs = await self._deduplicator.deduplicate(
                internal_docs, request, progress_callback=tracker.update_deduplication_progress
            )
            duplicates_removed = len(internal_docs) - len(deduplicated_docs)
            await tracker.complete_step(
                "deduplicating",
                f"Checked {len(internal_docs)} chunks, removed {duplicates_removed} duplicates"
            )

            if not deduplicated_docs:
                raise ParsingError("All chunks were duplicates")

            # Separate tables from regular docs
            table_docs, regular_docs = self._separate_tables(deduplicated_docs)

            # Write table documents
            if table_docs and self._table_retriever:
                success = await self._writer.write_table_documents(
                    table_docs, self._table_retriever
                )
                if not success:
                    regular_docs.extend(table_docs)
            elif table_docs:
                regular_docs.extend(table_docs)

            # Write regular documents
            if regular_docs:
                await tracker.start_step("embedding")
                await self._progress.update_state(operation_id, CheckpointState.EMBEDDING)

                await self._writer.write_documents(
                    regular_docs,
                    checkpoint_callback=(
                        lambda doc_id: self._progress.record_processed_chunk(
                            operation_id, doc_id
                        )
                    ),
                )

                await tracker.complete_step(
                    "embedding", f"Generated embeddings for {len(regular_docs)} chunks"
                )

                await tracker.start_step("storing")
                await self._progress.update_state(operation_id, CheckpointState.STORING)
                await tracker.complete_step(
                    "storing", f"Stored {len(regular_docs)} documents"
                )

            # Refresh indices
            await self._writer.refresh_bm25_corpus()
            await self._update_cooccurrence_index(deduplicated_docs)
            await self._update_source_catalog(deduplicated_docs)

            # Cache document info
            if self._cache_service:
                await self._cache_document_info(doc_id, deduplicated_docs, request)

            # Mark complete
            await self._progress.mark_completed(operation_id)

            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()

            self._performance_monitor.record_ingestion_result(
                processing_time_ms=processing_time * 1000,
                chunks=len(deduplicated_docs),
                invalid_chunks=invalid_chunk_count,
                status="success",
            )

            return DocumentIngestionResponse(
                document_id=doc_id,
                chunks_created=len(deduplicated_docs),
                status="success",
                message=f"Successfully ingested into {len(deduplicated_docs)} chunks",
                processing_time=processing_time,
                error_details=(
                    {
                        "original_chunks": len(internal_docs),
                        "deduplicated_chunks": len(deduplicated_docs),
                        "duplicates_removed": len(internal_docs) - len(deduplicated_docs),
                    }
                    if len(internal_docs) != len(deduplicated_docs)
                    else None
                ),
            )

        except (ParsingError, ValidationError) as e:
            return await self._handle_error(
                e, operation_id, start_time, deduplicated_docs, invalid_chunk_count
            )

        except Exception as e:
            from app.core.errors import categorize_error

            categorized = categorize_error(e)
            return await self._handle_error(
                categorized,
                operation_id,
                start_time,
                deduplicated_docs,
                invalid_chunk_count,
            )

        finally:
            self._performance_monitor.adjust_ingestion_in_progress(-1)

    def _validate_request(self, request: DocumentIngestionRequest) -> None:
        """Validate ingestion request."""
        if not request.url and not request.file_path and not request.content:
            raise ValidationError(
                "Must provide either URL, file path, or content",
                field="source",
            )

        if request.url:
            import re

            url_pattern = re.compile(
                r"^https?://"
                r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|"
                r"localhost|"
                r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"
                r"(?::\d+)?"
                r"(?:/?|[/?]\S+)$",
                re.IGNORECASE,
            )
            if not url_pattern.match(request.url):
                raise ValidationError(
                    f"Invalid URL format: {request.url}",
                    field="url",
                    value=request.url,
                )

        if request.file_path:
            import os

            if not os.path.exists(request.file_path):
                raise ValidationError(
                    f"File not found: {request.file_path}",
                    field="file_path",
                    value=request.file_path,
                )

        try:
            DocumentType(request.type)
        except ValueError:
            raise ValidationError(
                f"Invalid document type: {request.type}",
                field="type",
                value=request.type,
            )

    def _generate_document_id(self, request: DocumentIngestionRequest) -> str:
        """Generate unique document ID."""
        if request.content:
            content_hash = self._content_hasher.generate_content_hash(request.content)
            return f"doc_{content_hash[:12]}"

        source = request.url or request.file_path or str(uuid.uuid4())
        hash_input = f"{source}:{request.type}:{str(request.metadata)}"
        hash_obj = hashlib.sha256(hash_input.encode())
        return f"doc_{hash_obj.hexdigest()[:12]}"

    async def _check_existing(
        self,
        request: DocumentIngestionRequest,
    ) -> Optional[Dict[str, Any]]:
        """Check if document already exists."""
        if not self._cache_service:
            return None

        doc_id = self._generate_document_id(request)
        return await self._cache_service.get(f"doc:{doc_id}")

    def _create_exists_response(
        self,
        existing: Dict[str, Any],
        start_time: datetime,
    ) -> DocumentIngestionResponse:
        """Create response for already-existing document."""
        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()

        self._performance_monitor.record_ingestion_result(
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
            processing_time=0,
        )

    def _separate_tables(
        self,
        documents: List[Document],
    ) -> tuple[List[Document], List[Document]]:
        """Separate table documents from regular documents."""
        table_docs = []
        regular_docs = []

        table_types = {
            "table_markdown",
            "table_key_value",
            "table_html",
            "table_json",
            "table_unstructured",
        }

        for doc in documents:
            if hasattr(doc.metadata, "model_dump"):
                metadata = doc.metadata.model_dump()
            elif isinstance(doc.metadata, dict):
                metadata = doc.metadata
            else:
                metadata = {}

            content_type = metadata.get("content_type", "")
            if content_type in table_types:
                table_docs.append(doc)
            else:
                regular_docs.append(doc)

        logger.info(
            f"Separated: {len(table_docs)} tables, {len(regular_docs)} regular"
        )
        return table_docs, regular_docs

    async def _cache_document_info(
        self,
        doc_id: str,
        docs: List[Document],
        request: DocumentIngestionRequest,
    ) -> None:
        """Cache document information."""
        try:
            await self._cache_service.set(
                f"doc:{doc_id}",
                {
                    "id": doc_id,
                    "chunks": len(docs),
                    "source": request.url or request.file_path or "direct",
                    "type": request.type,
                    "ingested_at": datetime.now(timezone.utc).isoformat(),
                    "metadata": request.metadata,
                },
                ttl=86400,
            )
        except Exception as e:
            logger.warning(f"Failed to cache document info: {e}")

    async def _update_cooccurrence_index(self, documents: List[Document]) -> None:
        """Update co-occurrence index."""
        try:
            for doc in documents:
                if hasattr(doc.metadata, "model_dump"):
                    metadata = doc.metadata.model_dump()
                elif isinstance(doc.metadata, dict):
                    metadata = doc.metadata
                else:
                    metadata = {}

                metadata["id"] = doc.id

                langchain_doc = LangchainDocument(
                    page_content=doc.content,
                    metadata=metadata,
                )
                self._cooccurrence_indexer.add_document(langchain_doc)

            self._cooccurrence_indexer.save_index()
            logger.info(f"Updated co-occurrence index with {len(documents)} documents")

        except Exception as e:
            logger.error(f"Failed to update co-occurrence index: {e}")

    async def _update_source_catalog(self, documents: List[Document]) -> None:
        """Update source catalog with ingested documents."""
        if not self._source_repository or not documents:
            return

        try:
            entries: Dict[str, SourceCatalogEntry] = {}

            for doc in documents:
                metadata = (
                    doc.metadata.model_dump()
                    if hasattr(doc.metadata, "model_dump")
                    else dict(doc.metadata)
                )
                source_id = metadata.get("source_id")

                if not source_id:
                    continue

                entry = entries.get(source_id)
                if not entry:
                    entry = SourceCatalogEntry(
                        source_id=source_id,
                        title=metadata.get("title"),
                        canonical_url=(
                            metadata.get("canonical_url") or metadata.get("source")
                        ),
                        reference_path=(
                            metadata.get("reference_path") or metadata.get("section")
                        ),
                        document_type=(
                            str(metadata.get("type")) if metadata.get("type") else None
                        ),
                        section=metadata.get("section"),
                        metadata={
                            "source": metadata.get("source"),
                            "policy_reference": metadata.get("policy_reference"),
                            "page_number": metadata.get("page_number"),
                            "tags": metadata.get("tags"),
                        },
                    )
                    entries[source_id] = entry

                entry.last_ingested_at = datetime.now(timezone.utc)
                parent_ref = doc.parent_id or metadata.get("parent_id")
                entry.register_chunk(str(doc.id), str(parent_ref) if parent_ref else None)

            if entries:
                await self._source_repository.upsert_entries(entries.values())

        except Exception as e:
            logger.warning(f"Failed to update source repository: {e}")

    async def _handle_error(
        self,
        error: Exception,
        operation_id: str,
        start_time: datetime,
        deduplicated_docs: List[Document],
        invalid_chunk_count: int,
    ) -> DocumentIngestionResponse:
        """Handle ingestion error."""
        logger.error(f"Ingestion failed: {error}")

        await self._progress.mark_failed(operation_id, str(error))

        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()

        self._performance_monitor.record_ingestion_result(
            processing_time_ms=processing_time * 1000,
            chunks=len(deduplicated_docs),
            invalid_chunks=invalid_chunk_count,
            status="error",
        )

        error_dict = error.to_dict() if hasattr(error, "to_dict") else {"error": str(error)}
        message = error.message if hasattr(error, "message") else str(error)

        return DocumentIngestionResponse(
            document_id="",
            chunks_created=0,
            status="error",
            message=message,
            processing_time=processing_time,
            error_details=error_dict,
        )
