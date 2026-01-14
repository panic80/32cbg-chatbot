"""Result data models for evaluation."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set


@dataclass
class Claim:
    """Single factual claim extracted from an answer."""

    text: str
    source_sentence: str
    claim_type: str = "fact"  # fact, number, definition, procedure
    nli_label: Optional[str] = None  # entailment, neutral, contradiction
    nli_confidence: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "source_sentence": self.source_sentence,
            "claim_type": self.claim_type,
            "nli_label": self.nli_label,
            "nli_confidence": self.nli_confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Claim":
        return cls(
            text=data["text"],
            source_sentence=data["source_sentence"],
            claim_type=data.get("claim_type", "fact"),
            nli_label=data.get("nli_label"),
            nli_confidence=data.get("nli_confidence"),
        )


@dataclass
class RetrievalMetrics:
    """Retrieval quality metrics for a single query."""

    query: str
    retrieved_ids: List[str]
    relevant_ids: Set[str]
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    ndcg_at_k: Dict[int, float] = field(default_factory=dict)
    hit_rate_at_k: Dict[int, float] = field(default_factory=dict)
    latency_ms: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "retrieved_ids": self.retrieved_ids,
            "relevant_ids": list(self.relevant_ids),
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "mrr": self.mrr,
            "ndcg_at_k": self.ndcg_at_k,
            "hit_rate_at_k": self.hit_rate_at_k,
            "latency_ms": self.latency_ms,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetrievalMetrics":
        return cls(
            query=data["query"],
            retrieved_ids=data["retrieved_ids"],
            relevant_ids=set(data["relevant_ids"]),
            precision_at_k=data.get("precision_at_k", {}),
            recall_at_k=data.get("recall_at_k", {}),
            mrr=data.get("mrr", 0.0),
            ndcg_at_k=data.get("ndcg_at_k", {}),
            hit_rate_at_k=data.get("hit_rate_at_k", {}),
            latency_ms=data.get("latency_ms", 0.0),
            error=data.get("error"),
        )


@dataclass
class GenerationMetrics:
    """Generation quality metrics for a single query."""

    query: str
    answer: str
    sources: List[str]
    relevance_score: float = 0.0  # 0-1, LLM-judged
    completeness_score: float = 0.0  # 0-1, LLM-judged
    grounding_score: float = 0.0  # 0-1, % claims supported by sources
    latency_ms: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "sources": self.sources,
            "relevance_score": self.relevance_score,
            "completeness_score": self.completeness_score,
            "grounding_score": self.grounding_score,
            "latency_ms": self.latency_ms,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenerationMetrics":
        return cls(
            query=data["query"],
            answer=data["answer"],
            sources=data.get("sources", []),
            relevance_score=data.get("relevance_score", 0.0),
            completeness_score=data.get("completeness_score", 0.0),
            grounding_score=data.get("grounding_score", 0.0),
            latency_ms=data.get("latency_ms", 0.0),
            error=data.get("error"),
        )


@dataclass
class HallucinationResult:
    """Hallucination detection result for a single query."""

    query: str
    answer: str
    claims: List[Claim] = field(default_factory=list)
    entailed_count: int = 0
    neutral_count: int = 0
    contradicted_count: int = 0
    hallucination_score: float = 0.0  # 0-1, higher = more hallucination
    flagged_claims: List[Claim] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def total_claims(self) -> int:
        return len(self.claims)

    @property
    def is_hallucinated(self) -> bool:
        """Check if answer has significant hallucination."""
        return self.contradicted_count > 0 or self.hallucination_score > 0.3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "claims": [c.to_dict() for c in self.claims],
            "entailed_count": self.entailed_count,
            "neutral_count": self.neutral_count,
            "contradicted_count": self.contradicted_count,
            "hallucination_score": self.hallucination_score,
            "flagged_claims": [c.to_dict() for c in self.flagged_claims],
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HallucinationResult":
        return cls(
            query=data["query"],
            answer=data["answer"],
            claims=[Claim.from_dict(c) for c in data.get("claims", [])],
            entailed_count=data.get("entailed_count", 0),
            neutral_count=data.get("neutral_count", 0),
            contradicted_count=data.get("contradicted_count", 0),
            hallucination_score=data.get("hallucination_score", 0.0),
            flagged_claims=[Claim.from_dict(c) for c in data.get("flagged_claims", [])],
            error=data.get("error"),
        )


@dataclass
class AggregateMetrics:
    """Aggregated metrics across all queries."""

    # Retrieval aggregates
    mean_precision_at_k: Dict[int, float] = field(default_factory=dict)
    mean_recall_at_k: Dict[int, float] = field(default_factory=dict)
    mean_mrr: float = 0.0
    mean_ndcg_at_k: Dict[int, float] = field(default_factory=dict)
    mean_hit_rate_at_k: Dict[int, float] = field(default_factory=dict)
    mean_retrieval_latency_ms: float = 0.0

    # Generation aggregates
    mean_relevance_score: float = 0.0
    mean_completeness_score: float = 0.0
    mean_grounding_score: float = 0.0
    mean_generation_latency_ms: float = 0.0

    # Hallucination aggregates
    mean_hallucination_score: float = 0.0
    total_claims: int = 0
    total_entailed: int = 0
    total_neutral: int = 0
    total_contradicted: int = 0
    queries_with_hallucination: int = 0

    # Counts
    total_queries: int = 0
    retrieval_errors: int = 0
    generation_errors: int = 0
    hallucination_errors: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_precision_at_k": self.mean_precision_at_k,
            "mean_recall_at_k": self.mean_recall_at_k,
            "mean_mrr": self.mean_mrr,
            "mean_ndcg_at_k": self.mean_ndcg_at_k,
            "mean_hit_rate_at_k": self.mean_hit_rate_at_k,
            "mean_retrieval_latency_ms": self.mean_retrieval_latency_ms,
            "mean_relevance_score": self.mean_relevance_score,
            "mean_completeness_score": self.mean_completeness_score,
            "mean_grounding_score": self.mean_grounding_score,
            "mean_generation_latency_ms": self.mean_generation_latency_ms,
            "mean_hallucination_score": self.mean_hallucination_score,
            "total_claims": self.total_claims,
            "total_entailed": self.total_entailed,
            "total_neutral": self.total_neutral,
            "total_contradicted": self.total_contradicted,
            "queries_with_hallucination": self.queries_with_hallucination,
            "total_queries": self.total_queries,
            "retrieval_errors": self.retrieval_errors,
            "generation_errors": self.generation_errors,
            "hallucination_errors": self.hallucination_errors,
        }


@dataclass
class EvaluationResult:
    """Complete evaluation results."""

    timestamp: datetime = field(default_factory=datetime.now)
    config_name: str = "default"
    retrieval_metrics: List[RetrievalMetrics] = field(default_factory=list)
    generation_metrics: List[GenerationMetrics] = field(default_factory=list)
    hallucination_results: List[HallucinationResult] = field(default_factory=list)
    aggregate: AggregateMetrics = field(default_factory=AggregateMetrics)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "config_name": self.config_name,
            "retrieval_metrics": [m.to_dict() for m in self.retrieval_metrics],
            "generation_metrics": [m.to_dict() for m in self.generation_metrics],
            "hallucination_results": [r.to_dict() for r in self.hallucination_results],
            "aggregate": self.aggregate.to_dict(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationResult":
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            config_name=data.get("config_name", "default"),
            retrieval_metrics=[
                RetrievalMetrics.from_dict(m) for m in data.get("retrieval_metrics", [])
            ],
            generation_metrics=[
                GenerationMetrics.from_dict(m)
                for m in data.get("generation_metrics", [])
            ],
            hallucination_results=[
                HallucinationResult.from_dict(r)
                for r in data.get("hallucination_results", [])
            ],
            aggregate=AggregateMetrics(**data.get("aggregate", {})),
            metadata=data.get("metadata", {}),
        )
