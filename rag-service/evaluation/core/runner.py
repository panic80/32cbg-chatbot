"""Main evaluation orchestrator."""

import asyncio
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from evaluation.client.rag_client import RAGClient
from evaluation.core.config import EvaluationConfig
from evaluation.core.results import (
    EvaluationResult,
    GenerationMetrics,
    HallucinationResult,
    RetrievalMetrics,
)
from evaluation.datagen.dataset import EvaluationDataset
from evaluation.hallucination.detector import HallucinationDetector
from evaluation.metrics.aggregator import MetricsAggregator
from evaluation.metrics.generation import GenerationEvaluator
from evaluation.metrics.retrieval import RetrievalEvaluator


class EvaluationRunner:
    """Main orchestrator for RAG evaluation."""

    def __init__(
        self,
        config: EvaluationConfig,
        rag_client: RAGClient = None,
        dataset: EvaluationDataset = None,
    ):
        """Initialize evaluation runner.

        Args:
            config: Evaluation configuration
            rag_client: Optional pre-configured RAG client
            dataset: Optional evaluation dataset
        """
        self.config = config
        self.dataset = dataset

        # Per-request retrieval config overrides (used for comparison testing)
        self._retrieval_config: Optional[Dict[str, Any]] = None

        # Initialize client
        self.client = rag_client or RAGClient(
            base_url=config.rag_url,
            admin_token=config.admin_token,
            timeout=config.request_timeout,
        )

        # Initialize evaluators
        self.retrieval_evaluator = RetrievalEvaluator(config.k_values)
        self.generation_evaluator = GenerationEvaluator(
            judge_model=config.llm_judge_model,
            temperature=config.judge_temperature,
        )
        self.hallucination_detector = HallucinationDetector(
            nli_model=config.nli_model,
            nli_device=config.nli_device,
            nli_batch_size=config.nli_batch_size,
            contradiction_threshold=config.contradiction_threshold,
            neutral_weight=config.neutral_weight,
        )
        self.aggregator = MetricsAggregator(config.k_values)

    async def __aenter__(self):
        await self.client.connect()
        return self

    async def __aexit__(self, *args):
        await self.client.close()

    async def run_full_evaluation(
        self,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> EvaluationResult:
        """Run complete evaluation pipeline.

        Args:
            progress_callback: Optional callback(current, total, message)

        Returns:
            EvaluationResult with all metrics
        """
        if not self.dataset:
            raise ValueError("No dataset loaded for evaluation")

        total_steps = len(self.dataset) * 3  # retrieval + generation + hallucination
        current_step = 0

        # Run retrieval evaluation
        if progress_callback:
            progress_callback(current_step, total_steps, "Evaluating retrieval...")

        retrieval_metrics = await self.run_retrieval_evaluation(
            progress_callback=lambda c, t: progress_callback(
                c, total_steps, f"Retrieval: {c}/{t}"
            )
            if progress_callback
            else None
        )
        current_step += len(self.dataset)

        # Run generation evaluation
        if progress_callback:
            progress_callback(
                current_step, total_steps, "Evaluating generation quality..."
            )

        generation_metrics = await self.run_generation_evaluation(
            progress_callback=lambda c, t: progress_callback(
                current_step + c, total_steps, f"Generation: {c}/{t}"
            )
            if progress_callback
            else None
        )
        current_step += len(self.dataset)

        # Run hallucination detection
        if progress_callback:
            progress_callback(
                current_step, total_steps, "Detecting hallucinations..."
            )

        hallucination_results = await self.run_hallucination_detection(
            progress_callback=lambda c, t: progress_callback(
                current_step + c, total_steps, f"Hallucination: {c}/{t}"
            )
            if progress_callback
            else None
        )

        # Aggregate results
        aggregate = self.aggregator.aggregate(
            retrieval_metrics=retrieval_metrics,
            generation_metrics=generation_metrics,
            hallucination_results=hallucination_results,
        )

        return EvaluationResult(
            timestamp=datetime.now(),
            config_name=self.config.output_path or "evaluation",
            retrieval_metrics=retrieval_metrics,
            generation_metrics=generation_metrics,
            hallucination_results=hallucination_results,
            aggregate=aggregate,
            metadata={
                "dataset_size": len(self.dataset),
                "k_values": self.config.k_values,
                "nli_model": self.config.nli_model,
            },
        )

    async def run_retrieval_evaluation(
        self,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[RetrievalMetrics]:
        """Run retrieval metrics evaluation.

        Args:
            progress_callback: Optional callback(current, total)

        Returns:
            List of RetrievalMetrics
        """
        if not self.dataset:
            raise ValueError("No dataset loaded for evaluation")

        metrics = []
        questions = self.dataset.get_questions_with_ground_truth()
        total = len(questions)

        for i, item in enumerate(questions):
            question = item["question"]
            ground_truth = item["ground_truth"]

            # Use fast retrieval-only endpoint with config if configured (for comparison testing)
            # This uses the full pipeline but skips LLM generation for speed
            # Note: _retrieval_config can be {} for baseline, so we check if it's not None
            if self._retrieval_config is not None:
                result = await self.client.retrieve_only(
                    query=question,
                    retrieval_config=self._retrieval_config,
                )

                if result.error:
                    metrics.append(
                        RetrievalMetrics(
                            query=question,
                            retrieved_ids=[],
                            relevant_ids=ground_truth.relevant_chunk_ids,
                            latency_ms=result.latency_ms,
                            error=result.error,
                        )
                    )
                else:
                    # Evaluate retrieval using returned documents
                    metric = self.retrieval_evaluator.evaluate_query(
                        query=question,
                        retrieved_docs=result.documents,
                        relevant_ids=ground_truth.relevant_chunk_ids,
                        relevance_scores=ground_truth.relevance_scores,
                        latency_ms=result.latency_ms,
                    )
                    metrics.append(metric)
            else:
                # Use simple search endpoint (default)
                result = await self.client.search(
                    query=question,
                    k=max(self.config.k_values),
                )

                if result.error:
                    metrics.append(
                        RetrievalMetrics(
                            query=question,
                            retrieved_ids=[],
                            relevant_ids=ground_truth.relevant_chunk_ids,
                            latency_ms=result.latency_ms,
                            error=result.error,
                        )
                    )
                else:
                    # Evaluate retrieval
                    metric = self.retrieval_evaluator.evaluate_query(
                        query=question,
                        retrieved_docs=result.documents,
                        relevant_ids=ground_truth.relevant_chunk_ids,
                        relevance_scores=ground_truth.relevance_scores,
                        latency_ms=result.latency_ms,
                    )
                    metrics.append(metric)

            if progress_callback:
                progress_callback(i + 1, total)

            # Rate limiting
            await asyncio.sleep(1.0 / self.config.requests_per_second)

        return metrics

    async def run_generation_evaluation(
        self,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[GenerationMetrics]:
        """Run generation quality evaluation.

        Args:
            progress_callback: Optional callback(current, total)

        Returns:
            List of GenerationMetrics
        """
        if not self.dataset:
            raise ValueError("No dataset loaded for evaluation")

        metrics = []
        questions = self.dataset.get_questions_with_ground_truth()
        total = len(questions)

        for i, item in enumerate(questions):
            question = item["question"]
            expected_answer = item.get("expected_answer")

            # Get RAG response
            result = await self.client.chat(
                message=question,
                use_rag=True,
            )

            if result.error:
                metrics.append(
                    GenerationMetrics(
                        query=question,
                        answer="",
                        sources=[],
                        latency_ms=result.latency_ms,
                        error=result.error,
                    )
                )
            else:
                # Extract source contents (try multiple possible field names)
                sources = []
                for s in result.sources:
                    content = (
                        s.get("text") or
                        s.get("content") or
                        s.get("page_content") or
                        ""
                    )
                    if content:
                        sources.append(content)

                # Evaluate generation
                metric = await self.generation_evaluator.evaluate_answer(
                    question=question,
                    answer=result.answer,
                    sources=sources,
                    expected_answer=expected_answer,
                    latency_ms=result.latency_ms,
                )
                metrics.append(metric)

            if progress_callback:
                progress_callback(i + 1, total)

            # Rate limiting
            await asyncio.sleep(1.0 / self.config.requests_per_second)

        return metrics

    async def run_hallucination_detection(
        self,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[HallucinationResult]:
        """Run hallucination detection.

        Args:
            progress_callback: Optional callback(current, total)

        Returns:
            List of HallucinationResult
        """
        if not self.dataset:
            raise ValueError("No dataset loaded for evaluation")

        results = []
        questions = self.dataset.get_questions_with_ground_truth()
        total = len(questions)

        for i, item in enumerate(questions):
            question = item["question"]

            # Get RAG response
            chat_result = await self.client.chat(
                message=question,
                use_rag=True,
            )

            if chat_result.error:
                results.append(
                    HallucinationResult(
                        query=question,
                        answer="",
                        error=chat_result.error,
                    )
                )
            else:
                # Extract source contents (try multiple possible field names)
                sources = []
                for s in chat_result.sources:
                    content = (
                        s.get("text") or
                        s.get("content") or
                        s.get("page_content") or
                        ""
                    )
                    if content:
                        sources.append(content)

                # Detect hallucinations
                result = await self.hallucination_detector.detect(
                    query=question,
                    answer=chat_result.answer,
                    sources=sources,
                )
                results.append(result)

            if progress_callback:
                progress_callback(i + 1, total)

            # Rate limiting
            await asyncio.sleep(1.0 / self.config.requests_per_second)

        return results

    async def run_retrieval_only(
        self,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> EvaluationResult:
        """Run retrieval evaluation only.

        Args:
            progress_callback: Optional callback

        Returns:
            EvaluationResult with retrieval metrics only
        """
        retrieval_metrics = await self.run_retrieval_evaluation(progress_callback)

        aggregate = self.aggregator.aggregate(retrieval_metrics=retrieval_metrics)

        return EvaluationResult(
            timestamp=datetime.now(),
            config_name=self.config.output_path or "retrieval_evaluation",
            retrieval_metrics=retrieval_metrics,
            aggregate=aggregate,
            metadata={
                "evaluation_type": "retrieval_only",
                "dataset_size": len(self.dataset) if self.dataset else 0,
                "k_values": self.config.k_values,
            },
        )

    async def compare_configurations(
        self,
        configs: Dict[str, Dict[str, Any]],
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> Dict[str, EvaluationResult]:
        """Compare multiple retrieval configurations.

        Args:
            configs: Dict mapping config name to config values
            progress_callback: Optional callback(config_name, current, total)

        Returns:
            Dict mapping config name to EvaluationResult
        """
        if not self.dataset:
            raise ValueError("No dataset loaded for evaluation")

        results = {}

        for config_name, config_values in configs.items():
            if progress_callback:
                progress_callback(config_name, 0, len(self.dataset))

            # Set per-request retrieval config for this evaluation run
            # This uses the chat endpoint with retrieval_config to apply the settings
            # Even for baseline (empty config), we use the chat endpoint for consistency
            self._retrieval_config = config_values if config_values else {}
            if config_values:
                print(f"Config '{config_name}': {config_values}")
            else:
                print(f"Config '{config_name}': baseline (no overrides)")

            # Run evaluation with the configured retrieval settings
            result = await self.run_retrieval_only(
                progress_callback=lambda c, t: progress_callback(config_name, c, t)
                if progress_callback
                else None
            )
            result.config_name = config_name
            result.metadata["config_values"] = config_values

            results[config_name] = result

        # Reset retrieval config after comparison
        self._retrieval_config = None

        return results

    def load_dataset(self, path: str) -> None:
        """Load evaluation dataset from file.

        Args:
            path: Path to dataset JSON file
        """
        self.dataset = EvaluationDataset.load(path)
        print(f"Loaded dataset with {len(self.dataset)} questions")

    def set_dataset(self, dataset: EvaluationDataset) -> None:
        """Set evaluation dataset directly.

        Args:
            dataset: EvaluationDataset instance
        """
        self.dataset = dataset
