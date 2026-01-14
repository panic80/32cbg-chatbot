"""
RAG Evaluation Framework

Comprehensive evaluation suite for RAG systems including:
- Retrieval quality metrics (Precision@k, Recall@k, MRR, NDCG)
- Synthetic test data generation
- End-to-end generation quality assessment
- NLI-based hallucination detection
"""

from evaluation.core.config import EvaluationConfig
from evaluation.core.results import (
    RetrievalMetrics,
    GenerationMetrics,
    HallucinationResult,
    EvaluationResult,
)

__version__ = "0.1.0"

__all__ = [
    "EvaluationConfig",
    "RetrievalMetrics",
    "GenerationMetrics",
    "HallucinationResult",
    "EvaluationResult",
]
