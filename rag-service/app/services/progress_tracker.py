"""Progress tracking for ingestion and other long-running operations."""

import asyncio
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime, timezone
import json
from enum import Enum

from app.core.logging import get_logger
from app.core.config import settings

logger = get_logger(__name__)


class ProgressStatus(str, Enum):
    """Progress status types."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ERROR = "error"


class ProgressStep:
    """Represents a single step in a progress workflow."""
    
    def __init__(
        self,
        id: str,
        name: str,
        status: ProgressStatus = ProgressStatus.PENDING,
        progress: float = 0.0,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.id = id
        self.name = name
        self.status = status
        self.progress = progress
        self.message = message
        self.details = details or {}
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        
    def start(self, message: Optional[str] = None):
        """Mark step as started."""
        self.status = ProgressStatus.IN_PROGRESS
        self.start_time = datetime.now(timezone.utc)
        self.message = message or f"Starting {self.name}..."
        
    def update(self, progress: float, message: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """Update step progress."""
        self.progress = max(0, min(100, progress))
        if message:
            self.message = message
        if details:
            self.details.update(details)
            
    def complete(self, message: Optional[str] = None):
        """Mark step as completed."""
        self.status = ProgressStatus.COMPLETED
        self.progress = 100.0
        self.end_time = datetime.now(timezone.utc)
        self.message = message or f"{self.name} completed"
        
    def error(self, message: str):
        """Mark step as failed."""
        self.status = ProgressStatus.ERROR
        self.end_time = datetime.now(timezone.utc)
        self.message = message
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "details": self.details,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": (self.end_time - self.start_time).total_seconds() if self.start_time and self.end_time else None
        }


class ProgressTracker:
    """Tracks progress of multi-step operations."""
    
    def __init__(self, operation_id: str, steps: List[ProgressStep]):
        """Initialize progress tracker."""
        self.operation_id = operation_id
        self.steps = {step.id: step for step in steps}
        self.current_step_id: Optional[str] = None
        self.callbacks: List[Callable] = []
        self.overall_progress: float = 0.0
        self.start_time = datetime.now(timezone.utc)
        self.end_time: Optional[datetime] = None
        self.error: Optional[str] = None
        
    def add_callback(self, callback: Callable):
        """Add a progress update callback."""
        self.callbacks.append(callback)
        
    async def _notify_callbacks(self, event_type: str, data: Dict[str, Any]):
        """Notify all callbacks of progress update with timeout protection.

        Each callback is wrapped with a timeout to prevent slow callbacks
        from blocking the pipeline. Timeouts and errors are logged but don't
        stop other callbacks from being notified.
        """
        callback_timeout = settings.progress_callback_timeout

        for callback in self.callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    # Async callback with timeout
                    await asyncio.wait_for(
                        callback(event_type, data),
                        timeout=callback_timeout
                    )
                else:
                    # Sync callback - run in executor with timeout
                    loop = asyncio.get_event_loop()
                    await asyncio.wait_for(
                        loop.run_in_executor(None, callback, event_type, data),
                        timeout=callback_timeout
                    )
            except asyncio.TimeoutError:
                logger.warning(
                    f"Progress callback timed out after {callback_timeout}s for event: {event_type}"
                )
            except Exception as e:
                logger.error(f"Error in progress callback for {event_type}: {e}")
                
    async def start_step(self, step_id: str, message: Optional[str] = None):
        """Start a step."""
        if step_id not in self.steps:
            raise ValueError(f"Unknown step: {step_id}")
            
        step = self.steps[step_id]
        step.start(message)
        self.current_step_id = step_id
        
        await self._notify_callbacks("step_start", {
            "stepId": step_id,
            "message": step.message
        })
        
        self._update_overall_progress()
        
    async def update_step(
        self,
        step_id: str,
        progress: float,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Update step progress."""
        if step_id not in self.steps:
            raise ValueError(f"Unknown step: {step_id}")
            
        step = self.steps[step_id]
        step.update(progress, message, details)
        
        await self._notify_callbacks("step_progress", {
            "stepId": step_id,
            "progress": progress,
            "message": step.message,
            "details": details
        })
        
        self._update_overall_progress()
        
    async def complete_step(self, step_id: str, message: Optional[str] = None):
        """Complete a step."""
        if step_id not in self.steps:
            raise ValueError(f"Unknown step: {step_id}")
            
        step = self.steps[step_id]
        step.complete(message)
        
        await self._notify_callbacks("step_complete", {
            "stepId": step_id,
            "message": step.message
        })
        
        self._update_overall_progress()
        
        # Auto-advance to next pending step
        for next_step in self.steps.values():
            if next_step.status == ProgressStatus.PENDING:
                self.current_step_id = next_step.id
                break
        else:
            self.current_step_id = None
            
    async def error_step(self, step_id: str, message: str):
        """Mark step as failed."""
        if step_id not in self.steps:
            raise ValueError(f"Unknown step: {step_id}")
            
        step = self.steps[step_id]
        step.error(message)
        self.error = message
        
        await self._notify_callbacks("step_error", {
            "stepId": step_id,
            "message": message
        })
        
        await self._notify_callbacks("error", {
            "message": message
        })
        
    def _update_overall_progress(self):
        """Update overall progress based on step progress."""
        total_progress = sum(step.progress for step in self.steps.values())
        self.overall_progress = total_progress / len(self.steps)
        
        asyncio.create_task(self._notify_callbacks("overall_progress", {
            "progress": self.overall_progress
        }))
        
    async def complete(self):
        """Mark operation as complete."""
        self.end_time = datetime.now(timezone.utc)
        
        await self._notify_callbacks("complete", {
            "message": "Operation completed successfully",
            "duration": (self.end_time - self.start_time).total_seconds()
        })
        
    def get_status(self) -> Dict[str, Any]:
        """Get current status."""
        return {
            "operation_id": self.operation_id,
            "overall_progress": self.overall_progress,
            "current_step": self.current_step_id,
            "steps": {step_id: step.to_dict() for step_id, step in self.steps.items()},
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "error": self.error
        }


class IngestionProgressTracker(ProgressTracker):
    """Specialized progress tracker for document ingestion."""
    
    def __init__(self, operation_id: str, url: str):
        """Initialize ingestion progress tracker."""
        steps = [
            ProgressStep("loading", "Loading document"),
            ProgressStep("splitting", "Splitting into chunks"),
            ProgressStep("deduplicating", "Deduplicating chunks"),
            ProgressStep("embedding", "Generating embeddings"),
            ProgressStep("storing", "Storing in vector database")
        ]
        super().__init__(operation_id, steps)
        self.url = url
        self.document_size: Optional[int] = None
        self.chunk_count: Optional[int] = None
        self.embedding_count: Optional[int] = None
        self.dedup_count: Optional[int] = None
        
    async def update_loading_progress(self, bytes_loaded: int, total_bytes: Optional[int] = None):
        """Update loading progress."""
        if total_bytes:
            self.document_size = total_bytes
            progress = (bytes_loaded / total_bytes) * 100
            await self.update_step(
                "loading",
                progress,
                f"Loading document: {bytes_loaded}/{total_bytes} bytes",
                {"bytes_loaded": bytes_loaded, "total_bytes": total_bytes}
            )
        else:
            await self.update_step(
                "loading",
                50,  # Unknown size, show 50%
                f"Loading document: {bytes_loaded} bytes loaded",
                {"bytes_loaded": bytes_loaded}
            )
            
    async def update_splitting_progress(self, chunks_processed: int, total_chunks: Optional[int] = None):
        """Update splitting progress."""
        if total_chunks:
            self.chunk_count = total_chunks
            progress = (chunks_processed / total_chunks) * 100
            await self.update_step(
                "splitting",
                progress,
                f"Processing chunks: {chunks_processed}/{total_chunks}",
                {
                    "current": chunks_processed,
                    "total": total_chunks,
                    "rate": chunks_processed / max(1, (datetime.now(timezone.utc) - self.steps["splitting"].start_time).total_seconds()) if self.steps["splitting"].start_time else 0
                }
            )
        else:
            await self.update_step(
                "splitting",
                min(90, chunks_processed),  # Cap at 90% if total unknown
                f"Processing chunks: {chunks_processed} processed",
                {"current": chunks_processed}
            )

    async def update_deduplication_progress(self, chunks_checked: int, total_chunks: int, duplicates_found: int = 0):
        """Update deduplication progress."""
        self.dedup_count = total_chunks
        progress = (chunks_checked / total_chunks) * 100

        # Calculate rate
        elapsed = (datetime.now(timezone.utc) - self.steps["deduplicating"].start_time).total_seconds() if self.steps["deduplicating"].start_time else 1
        rate = chunks_checked / max(1, elapsed)

        await self.update_step(
            "deduplicating",
            progress,
            f"Checking duplicates: {chunks_checked}/{total_chunks}",
            {
                "current": chunks_checked,
                "total": total_chunks,
                "duplicates": duplicates_found,
                "rate": rate
            }
        )

    async def update_embedding_progress(self, embeddings_generated: int, total_embeddings: int):
        """Update embedding generation progress."""
        self.embedding_count = total_embeddings
        progress = (embeddings_generated / total_embeddings) * 100
        
        # Calculate rate
        elapsed = (datetime.now(timezone.utc) - self.steps["embedding"].start_time).total_seconds() if self.steps["embedding"].start_time else 1
        rate = embeddings_generated / max(1, elapsed)
        
        await self.update_step(
            "embedding",
            progress,
            f"Generating embeddings: {embeddings_generated}/{total_embeddings}",
            {
                "current": embeddings_generated,
                "total": total_embeddings,
                "rate": rate
            }
        )
        
    async def update_storing_progress(self, docs_stored: int, total_docs: int):
        """Update storing progress."""
        progress = (docs_stored / total_docs) * 100
        
        await self.update_step(
            "storing",
            progress,
            f"Storing documents: {docs_stored}/{total_docs}",
            {
                "current": docs_stored,
                "total": total_docs
            }
        )
