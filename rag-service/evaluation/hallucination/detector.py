"""Complete hallucination detection pipeline."""

import asyncio
from typing import List, Optional

from evaluation.core.results import Claim, HallucinationResult
from evaluation.hallucination.claim_extractor import ClaimExtractor
from evaluation.hallucination.nli_checker import NLIChecker


class HallucinationDetector:
    """Complete hallucination detection pipeline.

    Combines claim extraction with NLI-based verification to detect
    hallucinated content in generated answers.
    """

    def __init__(
        self,
        nli_model: str = "cross-encoder/nli-deberta-v3-small",
        nli_device: str = "cpu",
        nli_batch_size: int = 8,
        contradiction_threshold: float = 0.7,
        neutral_weight: float = 0.5,
        claim_extractor: Optional[ClaimExtractor] = None,
        nli_checker: Optional[NLIChecker] = None,
    ):
        """Initialize hallucination detector.

        Args:
            nli_model: Model name for NLI checking
            nli_device: Device for NLI model (cpu, cuda, mps)
            nli_batch_size: Batch size for NLI inference
            contradiction_threshold: Confidence threshold for flagging contradictions
            neutral_weight: Weight for neutral claims in hallucination score
            claim_extractor: Optional pre-configured ClaimExtractor
            nli_checker: Optional pre-configured NLIChecker
        """
        self.contradiction_threshold = contradiction_threshold
        self.neutral_weight = neutral_weight

        self.claim_extractor = claim_extractor or ClaimExtractor()
        self.nli_checker = nli_checker or NLIChecker(
            model_name=nli_model,
            device=nli_device,
            batch_size=nli_batch_size,
        )

    def is_available(self) -> bool:
        """Check if hallucination detection is available."""
        return self.nli_checker.is_available()

    async def detect(
        self,
        query: str,
        answer: str,
        sources: List[str],
    ) -> HallucinationResult:
        """Detect hallucinations in answer.

        Args:
            query: The original query
            answer: The generated answer to check
            sources: List of source document contents

        Returns:
            HallucinationResult with detection results
        """
        # Handle empty answer
        if not answer or not answer.strip():
            return HallucinationResult(
                query=query,
                answer=answer,
                claims=[],
                entailed_count=0,
                neutral_count=0,
                contradicted_count=0,
                hallucination_score=0.0,
                flagged_claims=[],
            )

        # Handle no sources (everything is potentially hallucinated)
        if not sources:
            claims = await self.claim_extractor.extract_claims(answer)
            return HallucinationResult(
                query=query,
                answer=answer,
                claims=claims,
                entailed_count=0,
                neutral_count=len(claims),
                contradicted_count=0,
                hallucination_score=1.0 if claims else 0.0,
                flagged_claims=claims,
                error="No sources provided for verification",
            )

        # Step 1: Extract claims
        claims = await self.claim_extractor.extract_claims(answer)

        if not claims:
            return HallucinationResult(
                query=query,
                answer=answer,
                claims=[],
                entailed_count=0,
                neutral_count=0,
                contradicted_count=0,
                hallucination_score=0.0,
                flagged_claims=[],
            )

        # Step 2: Check each claim against sources using NLI
        claim_texts = [c.text for c in claims]
        nli_results = self.nli_checker.check_claims_batch(claim_texts, sources)

        # Step 3: Aggregate results
        entailed = 0
        neutral = 0
        contradicted = 0
        flagged = []

        for claim, (label, confidence) in zip(claims, nli_results):
            # Update claim with NLI results
            claim.nli_label = label
            claim.nli_confidence = confidence

            if label == "entailment":
                entailed += 1
            elif label == "neutral":
                neutral += 1
            else:  # contradiction
                contradicted += 1
                if confidence >= self.contradiction_threshold:
                    flagged.append(claim)

        # Calculate hallucination score
        # Higher score = more hallucination
        total = len(claims)
        if total > 0:
            # Contradictions count fully, neutrals count partially
            hallucination_score = (
                contradicted + self.neutral_weight * neutral
            ) / total
        else:
            hallucination_score = 0.0

        return HallucinationResult(
            query=query,
            answer=answer,
            claims=claims,
            entailed_count=entailed,
            neutral_count=neutral,
            contradicted_count=contradicted,
            hallucination_score=hallucination_score,
            flagged_claims=flagged,
        )

    async def detect_batch(
        self,
        items: List[dict],
        progress_callback: callable = None,
    ) -> List[HallucinationResult]:
        """Detect hallucinations in multiple answers.

        Args:
            items: List of dicts with keys:
                - query: str
                - answer: str
                - sources: List[str]
            progress_callback: Optional callback(current, total)

        Returns:
            List of HallucinationResult objects
        """
        results = []
        total = len(items)

        for i, item in enumerate(items):
            result = await self.detect(
                query=item["query"],
                answer=item["answer"],
                sources=item.get("sources", []),
            )
            results.append(result)

            if progress_callback:
                progress_callback(i + 1, total)

            # Small delay between detections
            await asyncio.sleep(0.1)

        return results

    def get_summary(self, results: List[HallucinationResult]) -> dict:
        """Get summary statistics from multiple results.

        Args:
            results: List of HallucinationResult objects

        Returns:
            Summary dictionary
        """
        if not results:
            return {
                "total_queries": 0,
                "total_claims": 0,
                "entailed_claims": 0,
                "neutral_claims": 0,
                "contradicted_claims": 0,
                "queries_with_hallucination": 0,
                "mean_hallucination_score": 0.0,
            }

        total_claims = sum(r.total_claims for r in results)
        total_entailed = sum(r.entailed_count for r in results)
        total_neutral = sum(r.neutral_count for r in results)
        total_contradicted = sum(r.contradicted_count for r in results)
        queries_with_hallucination = sum(1 for r in results if r.is_hallucinated)
        mean_score = sum(r.hallucination_score for r in results) / len(results)

        return {
            "total_queries": len(results),
            "total_claims": total_claims,
            "entailed_claims": total_entailed,
            "neutral_claims": total_neutral,
            "contradicted_claims": total_contradicted,
            "queries_with_hallucination": queries_with_hallucination,
            "hallucination_rate": queries_with_hallucination / len(results),
            "mean_hallucination_score": round(mean_score, 4),
            "claim_support_rate": (
                total_entailed / total_claims if total_claims > 0 else 0.0
            ),
        }
