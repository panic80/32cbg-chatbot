"""Unit tests for the RRFMerger component."""

import pytest
from langchain_core.documents import Document

from app.components.rrf_merger import RRFMerger


def _doc(doc_id: str, content: str) -> Document:
    """Helper to create documents with stable identifiers."""
    return Document(page_content=content, metadata={"id": doc_id})


def _sample_results():
    """Build a representative set of retriever outputs."""
    doc_a = _doc("A", "travel policy section A")
    doc_b = _doc("B", "meal rate information for 2025")
    doc_c = _doc("C", "accommodation guidance and restrictions")

    return {
        "dense": [doc_a, doc_b, doc_c],
        "bm25": [doc_b, doc_c],
    }


def test_rrf_merger_applies_score_threshold():
    retriever_results = _sample_results()

    merger = RRFMerger(score_threshold=0.3)
    merged_docs, stats = merger.merge(retriever_results)

    kept_ids = [doc.document.metadata["id"] for doc in merged_docs]

    assert kept_ids == ["B", "C"]
    assert stats.filtered_below_threshold == 1
    assert all(doc.rrf_score >= 0.3 for doc in merged_docs)
    assert all("rrf_score" in doc.document.metadata for doc in merged_docs)
    assert all("rrf_rank" in doc.document.metadata for doc in merged_docs)


def test_rrf_merger_keeps_top_doc_when_all_below_threshold():
    retriever_results = _sample_results()

    merger = RRFMerger(normalize_scores=False, score_threshold=0.5)
    merged_docs, stats = merger.merge(retriever_results)

    assert len(merged_docs) == 1
    assert merged_docs[0].document.metadata["id"] == "B"
    assert stats.filtered_below_threshold == 2
    assert pytest.approx(merged_docs[0].rrf_score, rel=1e-5) == merged_docs[0].document.metadata["rrf_score"]
