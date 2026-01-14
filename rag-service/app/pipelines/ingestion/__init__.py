"""Modular ingestion pipeline package.

Provides a refactored, modular version of the document ingestion pipeline
with clear separation of concerns.

Components:
- DocumentLoader: Load documents from various sources (URL, file, content)
- DocumentChunker: Split documents into chunks with various strategies
- MetadataEnricher: Extract and enrich document metadata
- QualityValidator: Validate chunk quality
- Deduplicator: Detect and remove duplicate content
- VectorWriter: Write embeddings to vector store
- ProgressManager: Track ingestion progress
- IngestionOrchestrator: Main pipeline orchestrator
"""

from app.pipelines.ingestion.orchestrator import IngestionOrchestrator
from app.pipelines.ingestion.document_loader import DocumentLoader
from app.pipelines.ingestion.chunker import DocumentChunker
from app.pipelines.ingestion.metadata_enricher import IngestionMetadataEnricher
from app.pipelines.ingestion.deduplicator import Deduplicator
from app.pipelines.ingestion.vector_writer import VectorWriter
from app.pipelines.ingestion.progress_manager import ProgressManager

# Backwards compatibility alias
IngestionPipeline = IngestionOrchestrator

__all__ = [
    "IngestionOrchestrator",
    "IngestionPipeline",  # Alias for backwards compatibility
    "DocumentLoader",
    "DocumentChunker",
    "IngestionMetadataEnricher",
    "Deduplicator",
    "VectorWriter",
    "ProgressManager",
]
