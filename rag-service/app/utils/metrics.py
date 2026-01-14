"""Utility helpers for computing quality-focused RAG metrics."""

from __future__ import annotations

from statistics import mean, pstdev
from typing import Dict, Iterable, List, Optional, Tuple

from app.models.query import Source


def _count_tokens(text: Optional[str]) -> int:
    if not text:
        return 0
    return len(text.split())


def compute_quality_metrics(
    answer_text: str,
    sources: Iterable[Source],
    retrieval_results: Optional[Iterable[Tuple[object, Optional[float]]]] = None,
) -> Dict[str, float]:
    """Derive richer quality metrics for a RAG response."""
    metrics: Dict[str, float] = {}

    source_list = list(sources)

    answer_tokens = _count_tokens(answer_text)
    source_tokens = sum(_count_tokens(getattr(source, "text", None)) for source in source_list)
    metrics["answer_token_count"] = float(answer_tokens)
    metrics["source_token_count"] = float(source_tokens)
    metrics["source_count"] = float(len(source_list))

    if answer_tokens == 0:
        coverage_ratio = 1.0 if source_tokens > 0 else 0.0
    else:
        coverage_ratio = min(1.0, source_tokens / answer_tokens) if source_tokens > 0 else 0.0

    support_ratio = source_tokens / answer_tokens if answer_tokens > 0 else (float("inf") if source_tokens > 0 else 0.0)
    support_ratio = min(support_ratio, 5.0) if support_ratio not in (float("inf"), float("nan")) else 5.0

    hallucination_ratio = max(0.0, 1.0 - coverage_ratio)

    metrics["context_coverage_rate"] = coverage_ratio
    metrics["context_support_ratio"] = support_ratio
    metrics["hallucination_rate"] = hallucination_ratio
    metrics["answer_to_context_ratio"] = 0.0 if source_tokens == 0 else min(5.0, answer_tokens / max(1, source_tokens))

    score_values: List[float] = []
    if retrieval_results:
        for _, score in retrieval_results:
            if isinstance(score, (int, float)):
                score_values.append(float(score))

    if score_values:
        metrics["retrieval_score_avg"] = mean(score_values)
        metrics["retrieval_score_max"] = max(score_values)
        metrics["retrieval_score_min"] = min(score_values)
        metrics["retrieval_score_sum"] = sum(score_values)
        if len(score_values) > 1:
            metrics["retrieval_score_std"] = pstdev(score_values)
            top_two = sorted(score_values, reverse=True)[:2]
            if len(top_two) == 2:
                metrics["retrieval_score_gap"] = top_two[0] - top_two[1]

    return metrics
