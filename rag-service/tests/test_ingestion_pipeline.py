import asyncio
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, timezone

from app.core.vectorstore import VectorStoreManager
from app.pipelines.ingestion import IngestionPipeline
from app.models.documents import DocumentIngestionRequest, DocumentType
from app.services.cache import CacheService
from app.services.source_repository import SourceRepository
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings


class DummyEmbeddings(Embeddings):
    dimension = 32

    def embed_documents(self, texts):  # type: ignore[override]
        return [[float((idx % 5) + 1)] * self.dimension for idx, _ in enumerate(texts)]

    def embed_query(self, text):  # type: ignore[override]
        return [0.5] * self.dimension


@pytest.mark.asyncio
async def test_simple_ingestion_round_trip(tmp_path, monkeypatch):
    persist_dir = tmp_path / "chroma"
    persist_dir.mkdir()
    monkeypatch.setenv("RAG_CHROMA_PERSIST_DIRECTORY", str(persist_dir))

    from app.core.config import settings

    settings.chroma_persist_directory = str(persist_dir)
    settings.enable_metadata_extraction = False
    settings.enable_quality_validation = False

    vector_manager = VectorStoreManager()
    embeddings = DummyEmbeddings()
    vector_manager.embeddings = embeddings
    vector_manager.vector_store = Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )

    pipeline = IngestionPipeline(
        vector_store_manager=vector_manager,
        cache_service=None,
        source_repository=None,
        use_smart_chunking=False,
    )

    request = DocumentIngestionRequest(
        content="Meal allowances are $17.30 for the first 30 days.",
        type=DocumentType.TEXT,
        metadata={
            "source": "https://example.test/allowances",
            "title": "Allowances",
            "section": "Meals",
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    response = await pipeline.ingest_document(request)
    assert response.status == "success"
    assert response.chunks_created == 1

    stored_docs = vector_manager.get_all_documents(refresh=True)
    assert stored_docs, "Expected ingested chunk in temporary vector store"

    await pipeline.cleanup()


@pytest.mark.asyncio
async def test_checkpoint_serialization_limits(tmp_path, monkeypatch):
    from app.services.cache import CacheService

    cache_service = CacheService()
    cache_service.enabled = False  # operate without Redis

    persist_dir = tmp_path / "chroma"
    persist_dir.mkdir()
    monkeypatch.setenv("RAG_CHROMA_PERSIST_DIRECTORY", str(persist_dir))

    from app.core.config import settings

    settings.chroma_persist_directory = str(persist_dir)
    settings.enable_metadata_extraction = False
    settings.enable_quality_validation = False

    vector_manager = VectorStoreManager()
    embeddings = DummyEmbeddings()
    vector_manager.embeddings = embeddings
    vector_manager.vector_store = Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )

    pipeline = IngestionPipeline(
        vector_store_manager=vector_manager,
        cache_service=None,
        source_repository=None,
        use_smart_chunking=False,
    )

    long_text = "Line | Value\n" + "|".join(["col"] * 10) * 500
    request = DocumentIngestionRequest(
        content=long_text,
        type=DocumentType.TEXT,
        metadata={"source": "https://example.test/table"},
    )

    # Run ingestion; primarily validating serialization helper does not explode with long content
    response = await pipeline.ingest_document(request)
    assert response.status == "success"
    await pipeline.cleanup()
