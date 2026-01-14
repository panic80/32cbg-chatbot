"""Aggregate metrics across queries."""

from typing import Dict, List

from evaluation.core.results import (
    AggregateMetrics,
    GenerationMetrics,
    HallucinationResult,
    RetrievalMetrics,
)


class MetricsAggregator:
    """Aggregate evaluation metrics across multiple queries."""

    def __init__(self, k_values: List[int] = None):
        """Initialize aggregator.

        Args:
            k_values: List of k values for @k metrics
        """
        self.k_values = k_values or [1, 3, 5, 10]

    def aggregate(
        self,
        retrieval_metrics: List[RetrievalMetrics] = None,
        generation_metrics: List[GenerationMetrics] = None,
        hallucination_results: List[HallucinationResult] = None,
    ) -> AggregateMetrics:
        """Aggregate all metrics into summary statistics.

        Args:
            retrieval_metrics: List of retrieval metrics
            generation_metrics: List of generation metrics
            hallucination_results: List of hallucination results

        Returns:
            AggregateMetrics with computed averages
        """
        agg = AggregateMetrics()

        # Aggregate retrieval metrics
        if retrieval_metrics:
            valid_retrieval = [m for m in retrieval_metrics if m.error is None]
            agg.retrieval_errors = len(retrieval_metrics) - len(valid_retrieval)

            if valid_retrieval:
                agg = self._aggregate_retrieval(valid_retrieval, agg)

        # Aggregate generation metrics
        if generation_metrics:
            valid_generation = [m for m in generation_metrics if m.error is None]
            agg.generation_errors = len(generation_metrics) - len(valid_generation)

            if valid_generation:
                agg = self._aggregate_generation(valid_generation, agg)

        # Aggregate hallucination results
        if hallucination_results:
            valid_hallucination = [r for r in hallucination_results if r.error is None]
            agg.hallucination_errors = len(hallucination_results) - len(
                valid_hallucination
            )

            if valid_hallucination:
                agg = self._aggregate_hallucination(valid_hallucination, agg)

        # Set total queries
        agg.total_queries = max(
            len(retrieval_metrics or []),
            len(generation_metrics or []),
            len(hallucination_results or []),
        )

        return agg

    def _aggregate_retrieval(
        self,
        metrics: List[RetrievalMetrics],
        agg: AggregateMetrics,
    ) -> AggregateMetrics:
        """Aggregate retrieval metrics."""
        n = len(metrics)

        # Aggregate precision@k
        for k in self.k_values:
            values = [m.precision_at_k.get(k, 0.0) for m in metrics]
            agg.mean_precision_at_k[k] = sum(values) / n

        # Aggregate recall@k
        for k in self.k_values:
            values = [m.recall_at_k.get(k, 0.0) for m in metrics]
            agg.mean_recall_at_k[k] = sum(values) / n

        # Aggregate MRR
        agg.mean_mrr = sum(m.mrr for m in metrics) / n

        # Aggregate NDCG@k
        for k in self.k_values:
            values = [m.ndcg_at_k.get(k, 0.0) for m in metrics]
            agg.mean_ndcg_at_k[k] = sum(values) / n

        # Aggregate hit rate@k
        for k in self.k_values:
            values = [m.hit_rate_at_k.get(k, 0.0) for m in metrics]
            agg.mean_hit_rate_at_k[k] = sum(values) / n

        # Aggregate latency
        agg.mean_retrieval_latency_ms = sum(m.latency_ms for m in metrics) / n

        return agg

    def _aggregate_generation(
        self,
        metrics: List[GenerationMetrics],
        agg: AggregateMetrics,
    ) -> AggregateMetrics:
        """Aggregate generation metrics."""
        n = len(metrics)

        agg.mean_relevance_score = sum(m.relevance_score for m in metrics) / n
        agg.mean_completeness_score = sum(m.completeness_score for m in metrics) / n
        agg.mean_grounding_score = sum(m.grounding_score for m in metrics) / n
        agg.mean_generation_latency_ms = sum(m.latency_ms for m in metrics) / n

        return agg

    def _aggregate_hallucination(
        self,
        results: List[HallucinationResult],
        agg: AggregateMetrics,
    ) -> AggregateMetrics:
        """Aggregate hallucination results."""
        n = len(results)

        agg.mean_hallucination_score = sum(r.hallucination_score for r in results) / n
        agg.total_claims = sum(r.total_claims for r in results)
        agg.total_entailed = sum(r.entailed_count for r in results)
        agg.total_neutral = sum(r.neutral_count for r in results)
        agg.total_contradicted = sum(r.contradicted_count for r in results)
        agg.queries_with_hallucination = sum(1 for r in results if r.is_hallucinated)

        return agg

    def compute_retrieval_summary(
        self,
        metrics: List[RetrievalMetrics],
    ) -> Dict[str, float]:
        """Compute summary statistics for retrieval metrics.

        Args:
            metrics: List of retrieval metrics

        Returns:
            Dictionary with summary statistics
        """
        if not metrics:
            return {}

        valid = [m for m in metrics if m.error is None]
        if not valid:
            return {"error_rate": 1.0}

        n = len(valid)

        summary = {
            "total_queries": len(metrics),
            "successful_queries": n,
            "error_rate": (len(metrics) - n) / len(metrics),
            "mean_mrr": sum(m.mrr for m in valid) / n,
            "mean_latency_ms": sum(m.latency_ms for m in valid) / n,
        }

        # Add @k metrics for primary k value (usually 5)
        primary_k = 5 if 5 in self.k_values else self.k_values[0]
        summary[f"mean_precision@{primary_k}"] = (
            sum(m.precision_at_k.get(primary_k, 0) for m in valid) / n
        )
        summary[f"mean_recall@{primary_k}"] = (
            sum(m.recall_at_k.get(primary_k, 0) for m in valid) / n
        )
        summary[f"mean_hit_rate@{primary_k}"] = (
            sum(m.hit_rate_at_k.get(primary_k, 0) for m in valid) / n
        )

        return summary

    def compute_generation_summary(
        self,
        metrics: List[GenerationMetrics],
    ) -> Dict[str, float]:
        """Compute summary statistics for generation metrics.

        Args:
            metrics: List of generation metrics

        Returns:
            Dictionary with summary statistics
        """
        if not metrics:
            return {}

        valid = [m for m in metrics if m.error is None]
        if not valid:
            return {"error_rate": 1.0}

        n = len(valid)

        return {
            "total_queries": len(metrics),
            "successful_queries": n,
            "error_rate": (len(metrics) - n) / len(metrics),
            "mean_relevance": sum(m.relevance_score for m in valid) / n,
            "mean_completeness": sum(m.completeness_score for m in valid) / n,
            "mean_grounding": sum(m.grounding_score for m in valid) / n,
            "mean_latency_ms": sum(m.latency_ms for m in valid) / n,
        }

    def compute_hallucination_summary(
        self,
        results: List[HallucinationResult],
    ) -> Dict[str, float]:
        """Compute summary statistics for hallucination detection.

        Args:
            results: List of hallucination results

        Returns:
            Dictionary with summary statistics
        """
        if not results:
            return {}

        valid = [r for r in results if r.error is None]
        if not valid:
            return {"error_rate": 1.0}

        n = len(valid)
        total_claims = sum(r.total_claims for r in valid)

        summary = {
            "total_queries": len(results),
            "successful_queries": n,
            "error_rate": (len(results) - n) / len(results),
            "mean_hallucination_score": sum(r.hallucination_score for r in valid) / n,
            "total_claims": total_claims,
            "entailed_claims": sum(r.entailed_count for r in valid),
            "neutral_claims": sum(r.neutral_count for r in valid),
            "contradicted_claims": sum(r.contradicted_count for r in valid),
            "queries_with_hallucination": sum(1 for r in valid if r.is_hallucinated),
        }

        if total_claims > 0:
            summary["claim_support_rate"] = summary["entailed_claims"] / total_claims
            summary["claim_contradiction_rate"] = (
                summary["contradicted_claims"] / total_claims
            )

        return summary
