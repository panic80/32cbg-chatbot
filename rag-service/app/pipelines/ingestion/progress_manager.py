"""Progress tracking for ingestion pipeline."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document as LangchainDocument

from app.core.config import settings
from app.core.logging import get_logger
from app.services.progress_tracker import IngestionProgressTracker
from app.services.ingestion_checkpoint import (
    IngestionCheckpointService,
    CheckpointState,
)

logger = get_logger(__name__)


class ProgressManager:
    """Manages progress tracking and checkpointing for ingestion."""

    def __init__(
        self,
        cache_service: Optional[Any] = None,
    ):
        """Initialize progress manager.

        Args:
            cache_service: Cache service for checkpoint persistence.
        """
        self._checkpoint_service: Optional[IngestionCheckpointService] = None
        if cache_service:
            self._checkpoint_service = IngestionCheckpointService(cache_service)

        self._trackers: Dict[str, IngestionProgressTracker] = {}

    def create_tracker(
        self,
        operation_id: str,
        source: str,
        progress_callback: Optional[callable] = None,
    ) -> IngestionProgressTracker:
        """Create a progress tracker for an operation.

        Args:
            operation_id: Unique operation identifier.
            source: Document source description.
            progress_callback: Optional callback for progress updates.

        Returns:
            The progress tracker.
        """
        tracker = IngestionProgressTracker(operation_id, source)
        if progress_callback:
            tracker.add_callback(progress_callback)
        self._trackers[operation_id] = tracker
        return tracker

    def get_tracker(self, operation_id: str) -> Optional[IngestionProgressTracker]:
        """Get an existing tracker by operation ID."""
        return self._trackers.get(operation_id)

    async def check_resumable(
        self,
        operation_id: str,
    ) -> Optional[Any]:
        """Check if an operation can be resumed.

        Args:
            operation_id: The operation ID to check.

        Returns:
            Checkpoint if resumable, None otherwise.
        """
        if not self._checkpoint_service:
            return None

        checkpoint = await self._checkpoint_service.get_checkpoint(operation_id)
        if checkpoint and await self._checkpoint_service.can_resume(operation_id):
            logger.info(
                f"Resuming operation {operation_id} from state: {checkpoint.current_state}"
            )
            return checkpoint

        return None

    async def create_checkpoint(
        self,
        operation_id: str,
        document_source: str,
        total_documents: int = 0,
    ) -> Optional[Any]:
        """Create a new checkpoint.

        Args:
            operation_id: The operation ID.
            document_source: Source of the documents.
            total_documents: Total document count.

        Returns:
            The created checkpoint.
        """
        if not self._checkpoint_service:
            return None

        return await self._checkpoint_service.create_checkpoint(
            operation_id=operation_id,
            document_source=document_source,
            total_documents=total_documents,
        )

    async def update_state(
        self,
        operation_id: str,
        state: CheckpointState,
        metadata_update: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update checkpoint state.

        Args:
            operation_id: The operation ID.
            state: New state.
            metadata_update: Optional metadata to update.
        """
        if not self._checkpoint_service:
            return

        await self._checkpoint_service.update_progress(
            operation_id,
            new_state=state,
            metadata_update=metadata_update,
        )

    async def record_processed_chunk(
        self,
        operation_id: str,
        chunk_id: str,
    ) -> None:
        """Record a processed chunk for resume capability.

        Args:
            operation_id: The operation ID.
            chunk_id: The processed chunk ID.
        """
        if not self._checkpoint_service:
            return

        await self._checkpoint_service.update_progress(
            operation_id,
            processed_chunk_id=chunk_id,
        )

    async def mark_completed(self, operation_id: str) -> None:
        """Mark operation as completed."""
        if self._checkpoint_service:
            await self._checkpoint_service.mark_completed(operation_id)

        tracker = self._trackers.get(operation_id)
        if tracker:
            await tracker.complete()

    async def mark_failed(
        self,
        operation_id: str,
        error: str,
    ) -> None:
        """Mark operation as failed."""
        if self._checkpoint_service:
            await self._checkpoint_service.mark_failed(operation_id, error)

        tracker = self._trackers.get(operation_id)
        if tracker and tracker.current_step_id:
            await tracker.error_step(tracker.current_step_id, error)

    def serialize_documents(
        self,
        documents: List[LangchainDocument],
    ) -> List[Dict[str, Any]]:
        """Serialize documents for checkpoint storage.

        Args:
            documents: Documents to serialize.

        Returns:
            Serialized document list.
        """
        max_len = getattr(settings, "checkpoint_content_max_chars", 4000)
        serialized = []

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

    def deserialize_documents(
        self,
        serialized: List[Dict[str, Any]],
    ) -> List[LangchainDocument]:
        """Deserialize documents from checkpoint storage.

        Args:
            serialized: Serialized document list.

        Returns:
            Deserialized LangChain documents.
        """
        documents = []
        for item in serialized or []:
            documents.append(
                LangchainDocument(
                    page_content=item.get("page_content", ""),
                    metadata=item.get("metadata", {}),
                )
            )
        return documents
