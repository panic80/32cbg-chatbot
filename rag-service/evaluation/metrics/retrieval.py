"""Retrieval quality metrics: Precision@k, Recall@k, MRR, NDCG, Hit Rate."""

import math
from typing import Any, Dict, List, Set

from evaluation.core.results import RetrievalMetrics


class RetrievalEvaluator:
    """Compute retrieval quality metrics."""

    def __init__(self, k_values: List[int] = None):
        """Initialize evaluator.

        Args:
            k_values: List of k values for @k metrics (default: [1, 3, 5, 10])
        """
        self.k_values = k_values or [1, 3, 5, 10]

    def precision_at_k(
        self,
        retrieved_ids: List[str],
        relevant_ids: Set[str],
        k: int,
    ) -> float:
        """Compute Precision@k.

        Precision@k = (# relevant docs in top-k) / k

        Args:
            retrieved_ids: Ordered list of retrieved document IDs
            relevant_ids: Set of relevant document IDs (ground truth)
            k: Number of top results to consider

        Returns:
            Precision@k score (0.0 to 1.0)
        """
        if k <= 0:
            return 0.0

        retrieved_k = retrieved_ids[:k]
        relevant_retrieved = sum(1 for doc_id in retrieved_k if doc_id in relevant_ids)
        return relevant_retrieved / k

    def recall_at_k(
        self,
        retrieved_ids: List[str],
        relevant_ids: Set[str],
        k: int,
    ) -> float:
        """Compute Recall@k.

        Recall@k = (# relevant docs in top-k) / (# total relevant docs)

        Args:
            retrieved_ids: Ordered list of retrieved document IDs
            relevant_ids: Set of relevant document IDs (ground truth)
            k: Number of top results to consider

        Returns:
            Recall@k score (0.0 to 1.0)
        """
        if not relevant_ids:
            return 0.0

        retrieved_k = retrieved_ids[:k]
        relevant_retrieved = sum(1 for doc_id in retrieved_k if doc_id in relevant_ids)
        return relevant_retrieved / len(relevant_ids)

    def mean_reciprocal_rank(
        self,
        retrieved_ids: List[str],
        relevant_ids: Set[str],
    ) -> float:
        """Compute Mean Reciprocal Rank (MRR).

        MRR = 1 / (rank of first relevant document)

        Args:
            retrieved_ids: Ordered list of retrieved document IDs
            relevant_ids: Set of relevant document IDs (ground truth)

        Returns:
            MRR score (0.0 to 1.0)
        """
        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_ids:
                return 1.0 / rank
        return 0.0

    def ndcg_at_k(
        self,
        retrieved_ids: List[str],
        relevance_scores: Dict[str, float],
        k: int,
    ) -> float:
        """Compute Normalized Discounted Cumulative Gain (NDCG@k).

        NDCG@k = DCG@k / IDCG@k

        where DCG@k = sum(rel_i / log2(i + 1)) for i in 1..k

        Args:
            retrieved_ids: Ordered list of retrieved document IDs
            relevance_scores: Dict mapping doc_id to relevance score (0-3 typically)
            k: Number of top results to consider

        Returns:
            NDCG@k score (0.0 to 1.0)
        """
        if k <= 0:
            return 0.0

        dcg = self._dcg(retrieved_ids[:k], relevance_scores)

        # Compute ideal ranking
        ideal_ranking = sorted(
            relevance_scores.keys(),
            key=lambda x: relevance_scores.get(x, 0),
            reverse=True,
        )
        idcg = self._dcg(ideal_ranking[:k], relevance_scores)

        if idcg == 0:
            return 0.0

        return dcg / idcg

    def _dcg(self, ranking: List[str], relevance: Dict[str, float]) -> float:
        """Compute Discounted Cumulative Gain.

        Args:
            ranking: Ordered list of document IDs
            relevance: Dict mapping doc_id to relevance score

        Returns:
            DCG score
        """
        dcg = 0.0
        for i, doc_id in enumerate(ranking):
            rel = relevance.get(doc_id, 0)
            # Using log2(i + 2) because i is 0-indexed
            dcg += rel / math.log2(i + 2)
        return dcg

    def hit_rate_at_k(
        self,
        retrieved_ids: List[str],
        relevant_ids: Set[str],
        k: int,
    ) -> float:
        """Compute Hit Rate@k (binary success metric).

        Hit Rate@k = 1 if any relevant doc in top-k, else 0

        Args:
            retrieved_ids: Ordered list of retrieved document IDs
            relevant_ids: Set of relevant document IDs (ground truth)
            k: Number of top results to consider

        Returns:
            1.0 if hit, 0.0 if miss
        """
        if k <= 0 or not relevant_ids:
            return 0.0

        return 1.0 if any(doc_id in relevant_ids for doc_id in retrieved_ids[:k]) else 0.0

    def evaluate_query(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        relevant_ids: Set[str],
        relevance_scores: Dict[str, float] = None,
        latency_ms: float = 0.0,
    ) -> RetrievalMetrics:
        """Evaluate retrieval quality for a single query.

        Args:
            query: The query string
            retrieved_docs: List of retrieved documents with 'id' field
            relevant_ids: Set of relevant document IDs (ground truth)
            relevance_scores: Optional graded relevance scores for NDCG
            latency_ms: Query latency in milliseconds

        Returns:
            RetrievalMetrics with all computed metrics
        """
        # Extract IDs from retrieved docs
        retrieved_ids = []
        for doc in retrieved_docs:
            doc_id = doc.get("id") or doc.get("chunk_id") or doc.get("_id", "")
            if doc_id:
                retrieved_ids.append(str(doc_id))

        # Build relevance scores if not provided (binary: 1 for relevant, 0 for not)
        if relevance_scores is None:
            relevance_scores = {doc_id: 1.0 for doc_id in relevant_ids}

        # Compute all metrics
        precision_at_k = {}
        recall_at_k = {}
        ndcg_at_k = {}
        hit_rate_at_k = {}

        for k in self.k_values:
            precision_at_k[k] = self.precision_at_k(retrieved_ids, relevant_ids, k)
            recall_at_k[k] = self.recall_at_k(retrieved_ids, relevant_ids, k)
            ndcg_at_k[k] = self.ndcg_at_k(retrieved_ids, relevance_scores, k)
            hit_rate_at_k[k] = self.hit_rate_at_k(retrieved_ids, relevant_ids, k)

        mrr = self.mean_reciprocal_rank(retrieved_ids, relevant_ids)

        return RetrievalMetrics(
            query=query,
            retrieved_ids=retrieved_ids,
            relevant_ids=relevant_ids,
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            mrr=mrr,
            ndcg_at_k=ndcg_at_k,
            hit_rate_at_k=hit_rate_at_k,
            latency_ms=latency_ms,
        )

    def evaluate_batch(
        self,
        results: List[Dict[str, Any]],
    ) -> List[RetrievalMetrics]:
        """Evaluate multiple queries.

        Args:
            results: List of dicts with keys:
                - query: str
                - retrieved_docs: List[Dict]
                - relevant_ids: Set[str]
                - relevance_scores: Optional[Dict[str, float]]
                - latency_ms: float

        Returns:
            List of RetrievalMetrics
        """
        metrics = []
        for r in results:
            metrics.append(
                self.evaluate_query(
                    query=r["query"],
                    retrieved_docs=r["retrieved_docs"],
                    relevant_ids=r["relevant_ids"],
                    relevance_scores=r.get("relevance_scores"),
                    latency_ms=r.get("latency_ms", 0.0),
                )
            )
        return metrics
