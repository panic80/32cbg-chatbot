"""Generation quality metrics using LLM-as-judge."""

import asyncio
import json
import re
from typing import Any, List, Optional

from evaluation.core.results import GenerationMetrics


# Prompts for LLM-as-judge evaluation
RELEVANCE_PROMPT = """You are evaluating the relevance of an answer to a question.

Question: {question}

Answer: {answer}

Source documents used:
{sources}

Rate the answer's RELEVANCE to the question on a scale of 0-10:
- 0-2: Answer is completely off-topic or doesn't address the question
- 3-4: Answer partially addresses the question but misses key aspects
- 5-6: Answer addresses the question but lacks detail or precision
- 7-8: Answer directly addresses the question with good detail
- 9-10: Answer perfectly addresses the question comprehensively

Respond with JSON only:
{{"score": <0-10>, "reasoning": "<brief explanation>"}}"""


COMPLETENESS_PROMPT = """You are evaluating the completeness of an answer.

Question: {question}

Generated Answer: {answer}

Expected Answer (ground truth): {expected}

Rate the answer's COMPLETENESS compared to the expected answer on a scale of 0-10:
- 0-2: Answer misses most key information from expected answer
- 3-4: Answer includes some but misses significant information
- 5-6: Answer covers main points but lacks some details
- 7-8: Answer is mostly complete with minor omissions
- 9-10: Answer is fully complete, covering all expected information

Respond with JSON only:
{{"score": <0-10>, "reasoning": "<brief explanation>"}}"""


GROUNDING_PROMPT = """You are evaluating whether an answer is grounded in the provided sources.

Answer: {answer}

Source documents:
{sources}

Rate how well the answer is GROUNDED in the sources on a scale of 0-10:
- 0-2: Most claims are not supported by sources (likely hallucination)
- 3-4: Many claims lack source support
- 5-6: Some claims supported, some unsupported
- 7-8: Most claims are supported by sources
- 9-10: All claims are directly supported by sources

Respond with JSON only:
{{"score": <0-10>, "reasoning": "<brief explanation>", "unsupported_claims": ["<list any claims not in sources>"]}}"""


class GenerationEvaluator:
    """Evaluate answer quality using LLM-as-judge."""

    def __init__(
        self,
        llm_client: Any = None,
        judge_model: str = "gpt-4.1-mini",
        temperature: float = 0.0,
        max_concurrent: int = 5,
    ):
        """Initialize evaluator.

        Args:
            llm_client: Optional pre-configured LLM client
            judge_model: Model to use for judging (if llm_client not provided)
            temperature: Temperature for judge model
            max_concurrent: Maximum concurrent evaluations
        """
        self.llm = llm_client
        self.judge_model = judge_model
        self.temperature = temperature
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)

        if self.llm is None:
            self._init_default_llm()

    def _init_default_llm(self):
        """Initialize default LLM client."""
        try:
            from langchain_openai import ChatOpenAI

            self.llm = ChatOpenAI(
                model=self.judge_model,
                temperature=self.temperature,
            )
        except ImportError:
            print("Warning: langchain_openai not available, LLM evaluation disabled")
            self.llm = None

    async def _invoke_llm(self, prompt: str) -> str:
        """Invoke LLM with rate limiting."""
        if self.llm is None:
            return '{"score": 0, "reasoning": "LLM not available"}'

        async with self._semaphore:
            try:
                response = await self.llm.ainvoke(prompt)
                return response.content if hasattr(response, "content") else str(response)
            except Exception as e:
                return f'{{"score": 0, "reasoning": "Error: {str(e)}"}}'

    def _parse_score(self, response: str) -> float:
        """Parse score from LLM response.

        Args:
            response: Raw LLM response

        Returns:
            Score normalized to 0-1 range
        """
        try:
            # Try JSON parsing
            json_match = re.search(r"\{[\s\S]*\}", response)
            if json_match:
                data = json.loads(json_match.group())
                score = float(data.get("score", 0))
                return min(1.0, max(0.0, score / 10.0))

            # Fallback: look for number
            numbers = re.findall(r"(\d+(?:\.\d+)?)", response)
            if numbers:
                score = float(numbers[0])
                if score > 1:
                    score = score / 10.0
                return min(1.0, max(0.0, score))

        except (json.JSONDecodeError, ValueError):
            pass

        return 0.0

    async def evaluate_relevance(
        self,
        question: str,
        answer: str,
        sources: List[str],
    ) -> float:
        """Score answer relevance to question.

        Args:
            question: The original question
            answer: The generated answer
            sources: List of source document contents

        Returns:
            Relevance score (0.0 to 1.0)
        """
        sources_text = "\n---\n".join(sources[:5])  # Limit to 5 sources

        prompt = RELEVANCE_PROMPT.format(
            question=question,
            answer=answer,
            sources=sources_text,
        )

        response = await self._invoke_llm(prompt)
        return self._parse_score(response)

    async def evaluate_completeness(
        self,
        question: str,
        answer: str,
        expected_answer: Optional[str],
    ) -> float:
        """Score answer completeness.

        Args:
            question: The original question
            answer: The generated answer
            expected_answer: The expected/ground truth answer

        Returns:
            Completeness score (0.0 to 1.0)
        """
        if not expected_answer:
            # If no expected answer, give neutral score
            return 0.5

        prompt = COMPLETENESS_PROMPT.format(
            question=question,
            answer=answer,
            expected=expected_answer,
        )

        response = await self._invoke_llm(prompt)
        return self._parse_score(response)

    async def evaluate_grounding(
        self,
        answer: str,
        sources: List[str],
    ) -> float:
        """Score factual grounding in sources.

        Args:
            answer: The generated answer
            sources: List of source document contents

        Returns:
            Grounding score (0.0 to 1.0)
        """
        if not sources:
            return 0.0

        sources_text = "\n---\n".join(sources[:5])

        prompt = GROUNDING_PROMPT.format(
            answer=answer,
            sources=sources_text,
        )

        response = await self._invoke_llm(prompt)
        return self._parse_score(response)

    async def evaluate_answer(
        self,
        question: str,
        answer: str,
        sources: List[str],
        expected_answer: Optional[str] = None,
        latency_ms: float = 0.0,
    ) -> GenerationMetrics:
        """Complete answer evaluation.

        Args:
            question: The original question
            answer: The generated answer
            sources: List of source document contents
            expected_answer: Optional ground truth answer
            latency_ms: Answer generation latency

        Returns:
            GenerationMetrics with all scores
        """
        if not answer:
            return GenerationMetrics(
                query=question,
                answer=answer,
                sources=sources,
                relevance_score=0.0,
                completeness_score=0.0,
                grounding_score=0.0,
                latency_ms=latency_ms,
                error="Empty answer",
            )

        # Run evaluations in parallel
        relevance_task = self.evaluate_relevance(question, answer, sources)
        completeness_task = self.evaluate_completeness(question, answer, expected_answer)
        grounding_task = self.evaluate_grounding(answer, sources)

        relevance, completeness, grounding = await asyncio.gather(
            relevance_task,
            completeness_task,
            grounding_task,
        )

        return GenerationMetrics(
            query=question,
            answer=answer,
            sources=sources,
            relevance_score=relevance,
            completeness_score=completeness,
            grounding_score=grounding,
            latency_ms=latency_ms,
        )

    async def evaluate_batch(
        self,
        evaluations: List[dict],
        progress_callback: callable = None,
    ) -> List[GenerationMetrics]:
        """Evaluate multiple answers.

        Args:
            evaluations: List of dicts with keys:
                - question: str
                - answer: str
                - sources: List[str]
                - expected_answer: Optional[str]
                - latency_ms: float
            progress_callback: Optional callback(current, total)

        Returns:
            List of GenerationMetrics
        """
        results = []
        total = len(evaluations)

        for i, eval_data in enumerate(evaluations):
            metrics = await self.evaluate_answer(
                question=eval_data["question"],
                answer=eval_data["answer"],
                sources=eval_data.get("sources", []),
                expected_answer=eval_data.get("expected_answer"),
                latency_ms=eval_data.get("latency_ms", 0.0),
            )
            results.append(metrics)

            if progress_callback:
                progress_callback(i + 1, total)

            # Small delay between evaluations
            await asyncio.sleep(0.2)

        return results
