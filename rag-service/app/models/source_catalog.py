"""Models representing canonical source metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Set


@dataclass
class SourceCatalogEntry:
    """Aggregated metadata for a canonical source."""

    source_id: str
    title: Optional[str] = None
    canonical_url: Optional[str] = None
    reference_path: Optional[str] = None
    document_type: Optional[str] = None
    section: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_ingested_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    chunk_mappings: Dict[str, Optional[str]] = field(default_factory=dict)
    document_ids: Set[str] = field(default_factory=set)

    def register_chunk(self, chunk_id: str, document_id: Optional[str]) -> None:
        """Track a chunk that belongs to this source."""
        if not chunk_id:
            return
        self.chunk_mappings[chunk_id] = document_id
        if document_id:
            self.document_ids.add(document_id)

    @property
    def chunk_count(self) -> int:
        """Number of chunks currently registered."""
        return len(self.chunk_mappings)

    @property
    def document_count(self) -> int:
        """Number of unique parent documents associated with the source."""
        return len(self.document_ids)
