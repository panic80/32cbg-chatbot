"""Evaluation dataset management."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import random


@dataclass
class GeneratedQA:
    """Single Q&A pair with ground truth."""

    question: str
    answer: str
    question_type: str  # factual, procedural, comparison
    source_chunk_ids: List[str]
    source_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "question_type": self.question_type,
            "source_chunk_ids": self.source_chunk_ids,
            "source_content": self.source_content,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeneratedQA":
        return cls(
            question=data["question"],
            answer=data["answer"],
            question_type=data.get("question_type", "factual"),
            source_chunk_ids=data.get("source_chunk_ids", []),
            source_content=data.get("source_content", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass
class GroundTruth:
    """Ground truth for a query."""

    relevant_chunk_ids: Set[str]
    relevance_scores: Dict[str, float]  # chunk_id -> 0-3 relevance grade
    expected_answer: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "relevant_chunk_ids": list(self.relevant_chunk_ids),
            "relevance_scores": self.relevance_scores,
            "expected_answer": self.expected_answer,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GroundTruth":
        return cls(
            relevant_chunk_ids=set(data.get("relevant_chunk_ids", [])),
            relevance_scores=data.get("relevance_scores", {}),
            expected_answer=data.get("expected_answer"),
        )


class EvaluationDataset:
    """Manages evaluation dataset with Q&A pairs and ground truth."""

    def __init__(self, qa_pairs: List[GeneratedQA] = None):
        self.qa_pairs = qa_pairs or []
        self._index: Dict[str, int] = {}
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        """Rebuild question index for fast lookup."""
        self._index = {qa.question: i for i, qa in enumerate(self.qa_pairs)}

    def add(self, qa: GeneratedQA) -> None:
        """Add a Q&A pair to the dataset."""
        if qa.question in self._index:
            # Update existing
            self.qa_pairs[self._index[qa.question]] = qa
        else:
            self._index[qa.question] = len(self.qa_pairs)
            self.qa_pairs.append(qa)

    def add_batch(self, qa_pairs: List[GeneratedQA]) -> None:
        """Add multiple Q&A pairs."""
        for qa in qa_pairs:
            self.add(qa)

    def get(self, question: str) -> Optional[GeneratedQA]:
        """Get Q&A pair by question."""
        idx = self._index.get(question)
        if idx is not None:
            return self.qa_pairs[idx]
        return None

    def get_ground_truth(self, question: str) -> Optional[GroundTruth]:
        """Get ground truth for a query."""
        qa = self.get(question)
        if qa is None:
            return None

        return GroundTruth(
            relevant_chunk_ids=set(qa.source_chunk_ids),
            relevance_scores={cid: 1.0 for cid in qa.source_chunk_ids},
            expected_answer=qa.answer,
        )

    def filter_by_type(self, question_type: str) -> "EvaluationDataset":
        """Filter to specific question type.

        Args:
            question_type: One of 'factual', 'procedural', 'comparison'

        Returns:
            New dataset with only matching questions
        """
        filtered = [qa for qa in self.qa_pairs if qa.question_type == question_type]
        return EvaluationDataset(filtered)

    def sample(self, n: int, seed: int = None) -> "EvaluationDataset":
        """Random sample of n questions.

        Args:
            n: Number of questions to sample
            seed: Random seed for reproducibility

        Returns:
            New dataset with sampled questions
        """
        if seed is not None:
            random.seed(seed)

        n = min(n, len(self.qa_pairs))
        sampled = random.sample(self.qa_pairs, n)
        return EvaluationDataset(sampled)

    def stratified_sample(
        self,
        n_per_type: int,
        seed: int = None,
    ) -> "EvaluationDataset":
        """Stratified sample with equal representation per question type.

        Args:
            n_per_type: Number of questions per type
            seed: Random seed for reproducibility

        Returns:
            New dataset with stratified sample
        """
        if seed is not None:
            random.seed(seed)

        sampled = []
        for qtype in ["factual", "procedural", "comparison"]:
            type_qs = [qa for qa in self.qa_pairs if qa.question_type == qtype]
            n = min(n_per_type, len(type_qs))
            sampled.extend(random.sample(type_qs, n))

        return EvaluationDataset(sampled)

    def save(self, path: str) -> None:
        """Save dataset to JSON file.

        Args:
            path: Output file path
        """
        data = {
            "version": "1.0",
            "total_questions": len(self.qa_pairs),
            "question_types": self.get_type_counts(),
            "qa_pairs": [qa.to_dict() for qa in self.qa_pairs],
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "EvaluationDataset":
        """Load dataset from JSON file.

        Args:
            path: Input file path

        Returns:
            Loaded EvaluationDataset
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        qa_pairs = [GeneratedQA.from_dict(qa) for qa in data.get("qa_pairs", [])]
        return cls(qa_pairs)

    def get_type_counts(self) -> Dict[str, int]:
        """Get count of questions by type."""
        counts = {"factual": 0, "procedural": 0, "comparison": 0}
        for qa in self.qa_pairs:
            if qa.question_type in counts:
                counts[qa.question_type] += 1
        return counts

    def get_questions(self) -> List[str]:
        """Get all questions."""
        return [qa.question for qa in self.qa_pairs]

    def get_questions_with_ground_truth(
        self,
    ) -> List[Dict[str, Any]]:
        """Get questions with their ground truth for evaluation.

        Returns:
            List of dicts with 'question', 'ground_truth', 'expected_answer'
        """
        results = []
        for qa in self.qa_pairs:
            results.append(
                {
                    "question": qa.question,
                    "ground_truth": GroundTruth(
                        relevant_chunk_ids=set(qa.source_chunk_ids),
                        relevance_scores={cid: 1.0 for cid in qa.source_chunk_ids},
                        expected_answer=qa.answer,
                    ),
                    "expected_answer": qa.answer,
                    "question_type": qa.question_type,
                }
            )
        return results

    def __len__(self) -> int:
        return len(self.qa_pairs)

    def __iter__(self):
        return iter(self.qa_pairs)

    def __getitem__(self, idx: int) -> GeneratedQA:
        return self.qa_pairs[idx]

    def stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        type_counts = self.get_type_counts()

        # Calculate average answer length
        avg_answer_len = 0
        if self.qa_pairs:
            avg_answer_len = sum(len(qa.answer) for qa in self.qa_pairs) / len(
                self.qa_pairs
            )

        # Calculate average chunks per question
        avg_chunks = 0
        if self.qa_pairs:
            avg_chunks = sum(
                len(qa.source_chunk_ids) for qa in self.qa_pairs
            ) / len(self.qa_pairs)

        return {
            "total_questions": len(self.qa_pairs),
            "type_counts": type_counts,
            "avg_answer_length": round(avg_answer_len, 1),
            "avg_source_chunks": round(avg_chunks, 2),
            "unique_chunks": len(
                set(cid for qa in self.qa_pairs for cid in qa.source_chunk_ids)
            ),
        }
