"""Evaluation configuration."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EvaluationConfig:
    """Configuration for evaluation runs."""

    # RAG service connection
    rag_url: str = "http://localhost:8000"
    admin_token: str = ""
    request_timeout: float = 120.0

    # Retrieval metrics config
    k_values: List[int] = field(default_factory=lambda: [1, 3, 5, 10])

    # Synthetic data generation config
    questions_per_domain: int = 50
    question_types: List[str] = field(
        default_factory=lambda: ["factual", "procedural", "comparison"]
    )
    questions_per_chunk: int = 2
    generator_model: str = "gpt-4.1-mini"

    # NLI hallucination detection config
    nli_model: str = "cross-encoder/nli-deberta-v3-small"
    nli_device: str = "cpu"  # cpu, cuda, mps
    nli_batch_size: int = 8
    contradiction_threshold: float = 0.7
    neutral_weight: float = 0.25  # Weight for neutral claims in hallucination score

    # Generation quality config (LLM-as-judge)
    llm_judge_model: str = "gpt-4.1-mini"
    judge_temperature: float = 0.0

    # Output config
    output_format: str = "json"  # json, csv, markdown
    output_path: Optional[str] = None
    verbose: bool = False

    # Rate limiting
    requests_per_second: float = 5.0
    max_concurrent_requests: int = 5

    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []

        if not self.rag_url:
            errors.append("rag_url is required")

        if not self.k_values:
            errors.append("k_values must not be empty")
        elif any(k <= 0 for k in self.k_values):
            errors.append("all k_values must be positive")

        if self.questions_per_domain <= 0:
            errors.append("questions_per_domain must be positive")

        if not self.question_types:
            errors.append("question_types must not be empty")

        valid_question_types = {"factual", "procedural", "comparison"}
        invalid_types = set(self.question_types) - valid_question_types
        if invalid_types:
            errors.append(f"invalid question_types: {invalid_types}")

        if self.contradiction_threshold < 0 or self.contradiction_threshold > 1:
            errors.append("contradiction_threshold must be between 0 and 1")

        if self.neutral_weight < 0 or self.neutral_weight > 1:
            errors.append("neutral_weight must be between 0 and 1")

        valid_formats = {"json", "csv", "markdown"}
        if self.output_format not in valid_formats:
            errors.append(f"output_format must be one of {valid_formats}")

        return errors

    def is_valid(self) -> bool:
        """Check if configuration is valid."""
        return len(self.validate()) == 0
