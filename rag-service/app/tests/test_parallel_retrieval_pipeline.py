"""Tests for the ParallelRetrievalPipeline."""

import pytest
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.components.rrf_merger import RRFMerger
from app.pipelines.parallel_retrieval import ParallelRetrievalPipeline


class StaticRetriever(BaseRetriever):
    """Simple retriever that always returns the provided documents."""

    def __init__(self, documents):
        super().__init__()
        self._documents = documents

    def _get_relevant_documents(self, query, *, run_manager=None):
        return self._documents

    async def _aget_relevant_documents(self, query, *, run_manager=None):
        return self._documents


def _doc(doc_id: str, content: str) -> Document:
    return Document(page_content=content, metadata={"id": doc_id})


@pytest.mark.asyncio
async def test_parallel_pipeline_applies_rrf_threshold():
    doc_a = _doc("A", "travel policy section A")
    doc_b = _doc("B", "meal rate information for 2025")
    doc_c = _doc("C", "accommodation guidance and restrictions")

    retrievers = {
        "dense": StaticRetriever([doc_a, doc_b, doc_c]),
        "bm25": StaticRetriever([doc_b, doc_c]),
    }

    pipeline = ParallelRetrievalPipeline(
        retrievers=retrievers,
        rrf_merger=RRFMerger(score_threshold=0.3),
        concurrency_limit=2,
    )

    results = await pipeline.retrieve("meal rates", k=5)
    kept_ids = [doc.metadata["id"] for doc, _ in results]

    assert kept_ids == ["B", "C"]
    assert all("rrf_score" in doc.metadata for doc, _ in results)
