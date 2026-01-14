"""
Conditional Reranker with Fallback for intelligent result reranking.

This component decides when to apply reranking based on query analysis and result
characteristics, falling back to original RRF scores when reranking is not beneficial.

Reranking strategy:
- Apply reranking for complex, high-uncertainty queries where precision gains matter
- Skip reranking for simple queries or when result quality is already high
- Fallback to RRF scores when reranker fails or produces lower-quality results
- Dynamic confidence scoring to blend reranker and RRF scores optimally
"""

import time
import asyncio
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

from langchain_core.documents import Document

from app.components.rrf_merger import RRFDocument
from app.components.uncertainty_scorer import UncertaintyResult
from app.core.logging import get_logger

logger = get_logger(__name__)


class RerankingStrategy(Enum):
    """Reranking strategy decisions."""
    SKIP = "skip"           # Skip reranking entirely
    APPLY = "apply"         # Apply reranking
    BLEND = "blend"         # Blend reranker and RRF scores
    FALLBACK = "fallback"   # Use RRF scores due to reranker failure


@dataclass
class RerankingDecision:
    """Decision on whether and how to apply reranking."""
    strategy: RerankingStrategy
    confidence: float  # 0.0 = no confidence, 1.0 = full confidence
    reasoning: str
    
    # Decision factors
    query_complexity: float
    result_quality_score: float
    uncertainty_threshold_met: bool
    diversity_score: float
    
    # Performance considerations
    estimated_latency_ms: float
    skip_due_to_timeout: bool = False


@dataclass
class RerankingResult:
    """Result from conditional reranking operation."""
    documents: List[Document]
    strategy_used: RerankingStrategy
    reranker_confidence: float
    reranking_time_ms: float
    
    # Optional fields with defaults
    blend_ratio: Optional[float] = None  # RRF weight when blending
    fallback_triggered: bool = False
    fallback_reason: Optional[str] = None
    score_improvement: Optional[float] = None
    ranking_changes: int = 0  # Number of position changes


@dataclass
class ConditionalRerankerStats:
    """Statistics from conditional reranking."""
    total_requests: int = 0
    strategy_counts: Dict[str, int] = field(default_factory=dict)
    average_reranking_time_ms: float = 0.0
    fallback_rate: float = 0.0
    average_confidence: float = 0.0
    
    def update_strategy_count(self, strategy: RerankingStrategy):
        """Update strategy usage counts."""
        strategy_name = strategy.value
        if strategy_name not in self.strategy_counts:
            self.strategy_counts[strategy_name] = 0
        self.strategy_counts[strategy_name] += 1
        self.total_requests += 1


class ConditionalReranker:
    """
    Intelligent reranker that conditionally applies reranking based on query analysis.
    
    This component analyzes queries and results to determine the optimal reranking
    strategy, balancing precision gains against latency costs.
    """
    
    def __init__(
        self,
        reranker_function: Optional[callable] = None,
        uncertainty_threshold: float = 0.6,      # Minimum uncertainty for reranking
        min_documents_for_reranking: int = 5,    # Minimum docs to justify reranking
        max_reranking_latency_ms: float = 2000.0, # Maximum acceptable reranking time
        confidence_threshold: float = 0.7,       # Minimum confidence for pure reranking
        blend_threshold: float = 0.4,            # Threshold for blending vs pure strategies
        enable_fallback: bool = True,            # Enable fallback to RRF scores
        quality_improvement_threshold: float = 0.05  # Minimum improvement to keep reranker results
    ):
        """
        Initialize conditional reranker.
        
        Args:
            reranker_function: Async function that reranks documents
            uncertainty_threshold: Minimum query uncertainty to trigger reranking
            min_documents_for_reranking: Minimum documents needed for reranking
            max_reranking_latency_ms: Maximum time to spend on reranking
            confidence_threshold: Confidence threshold for pure reranking strategy
            blend_threshold: Confidence threshold for blending strategies
            enable_fallback: Whether to fallback to RRF on reranker failures
            quality_improvement_threshold: Minimum score improvement to accept reranking
        """
        self.reranker_function = reranker_function
        self.uncertainty_threshold = uncertainty_threshold
        self.min_documents_for_reranking = min_documents_for_reranking
        self.max_reranking_latency_ms = max_reranking_latency_ms
        self.confidence_threshold = confidence_threshold
        self.blend_threshold = blend_threshold
        self.enable_fallback = enable_fallback
        self.quality_improvement_threshold = quality_improvement_threshold
        
        # Statistics tracking
        self.stats = ConditionalRerankerStats()
        
        logger.info(f"ConditionalReranker initialized: uncertainty_threshold={uncertainty_threshold}, "
                   f"min_docs={min_documents_for_reranking}, max_latency={max_reranking_latency_ms}ms")
    
    async def rerank(
        self,
        query: str,
        documents: List[RRFDocument],
        uncertainty_result: Optional[UncertaintyResult] = None,
        max_results: Optional[int] = None
    ) -> Tuple[List[Document], RerankingResult]:
        """
        Conditionally rerank documents based on query analysis.
        
        Args:
            query: Original search query
            documents: RRF-scored documents to potentially rerank
            uncertainty_result: Query uncertainty analysis
            max_results: Maximum number of results to return
            
        Returns:
            Tuple of (reranked_documents, reranking_result)
        """
        start_time = time.time()
        
        # Phase 1: Make reranking decision
        decision = self._make_reranking_decision(query, documents, uncertainty_result)
        
        logger.debug(f"Reranking decision: {decision.strategy.value} (confidence: {decision.confidence:.2f}) - {decision.reasoning}")
        
        # Phase 2: Execute strategy
        if decision.strategy == RerankingStrategy.SKIP:
            result = await self._execute_skip_strategy(documents, decision, max_results)
        elif decision.strategy == RerankingStrategy.APPLY:
            result = await self._execute_apply_strategy(query, documents, decision, max_results)
        elif decision.strategy == RerankingStrategy.BLEND:
            result = await self._execute_blend_strategy(query, documents, decision, max_results)
        else:  # FALLBACK
            result = await self._execute_fallback_strategy(documents, decision, max_results)
        
        # Phase 3: Update statistics
        total_time = (time.time() - start_time) * 1000
        result.reranking_time_ms = total_time
        self.stats.update_strategy_count(decision.strategy)
        self._update_running_stats(result)
        
        logger.debug(f"Conditional reranking completed: {len(result.documents)} docs, "
                    f"strategy={result.strategy_used.value}, time={total_time:.1f}ms")
        
        return result.documents, result
    
    def _make_reranking_decision(
        self,
        query: str,
        documents: List[RRFDocument],
        uncertainty_result: Optional[UncertaintyResult] = None
    ) -> RerankingDecision:
        """Make intelligent decision about reranking strategy."""
        
        # Check basic prerequisites
        if not self.reranker_function:
            return RerankingDecision(
                strategy=RerankingStrategy.SKIP,
                confidence=1.0,
                reasoning="No reranker function available",
                query_complexity=0.0,
                result_quality_score=0.0,
                uncertainty_threshold_met=False,
                diversity_score=0.0,
                estimated_latency_ms=0.0
            )
        
        if len(documents) < self.min_documents_for_reranking:
            return RerankingDecision(
                strategy=RerankingStrategy.SKIP,
                confidence=1.0,
                reasoning=f"Too few documents ({len(documents)} < {self.min_documents_for_reranking})",
                query_complexity=0.0,
                result_quality_score=0.0,
                uncertainty_threshold_met=False,
                diversity_score=0.0,
                estimated_latency_ms=0.0
            )
        
        # Analyze query complexity and uncertainty
        query_complexity = self._calculate_query_complexity(query)
        uncertainty_score = uncertainty_result.overall_uncertainty if uncertainty_result else 0.5
        uncertainty_threshold_met = uncertainty_score >= self.uncertainty_threshold
        
        # Analyze result quality
        result_quality = self._analyze_result_quality(documents)
        diversity_score = self._calculate_diversity_score(documents)
        
        # Estimate reranking latency
        estimated_latency = self._estimate_reranking_latency(len(documents))
        skip_due_to_timeout = estimated_latency > self.max_reranking_latency_ms
        
        # Calculate overall confidence in reranking
        confidence = self._calculate_reranking_confidence(
            query_complexity, uncertainty_score, result_quality, diversity_score
        )
        
        # Make strategy decision
        if skip_due_to_timeout:
            strategy = RerankingStrategy.SKIP
            reasoning = f"Estimated latency too high ({estimated_latency:.0f}ms > {self.max_reranking_latency_ms}ms)"
        elif not uncertainty_threshold_met and result_quality > 0.8:
            strategy = RerankingStrategy.SKIP
            reasoning = f"Low uncertainty ({uncertainty_score:.2f}) and high result quality ({result_quality:.2f})"
        elif confidence >= self.confidence_threshold:
            strategy = RerankingStrategy.APPLY
            reasoning = f"High confidence ({confidence:.2f}) in reranking benefits"
        elif confidence >= self.blend_threshold:
            strategy = RerankingStrategy.BLEND
            reasoning = f"Moderate confidence ({confidence:.2f}) - blending RRF and reranker scores"
        else:
            strategy = RerankingStrategy.SKIP
            reasoning = f"Low confidence ({confidence:.2f}) in reranking benefits"
        
        return RerankingDecision(
            strategy=strategy,
            confidence=confidence,
            reasoning=reasoning,
            query_complexity=query_complexity,
            result_quality_score=result_quality,
            uncertainty_threshold_met=uncertainty_threshold_met,
            diversity_score=diversity_score,
            estimated_latency_ms=estimated_latency,
            skip_due_to_timeout=skip_due_to_timeout
        )
    
    def _calculate_query_complexity(self, query: str) -> float:
        """Calculate query complexity score (0.0 to 1.0)."""
        complexity = 0.0
        
        # Length factor
        word_count = len(query.split())
        complexity += min(word_count / 10.0, 0.3)
        
        # Technical terms
        technical_indicators = [
            'form', 'dnd', 'regulation', 'policy', 'procedure', 'section',
            'allowance', 'entitlement', 'authorization', 'reimbursement'
        ]
        tech_count = sum(1 for term in technical_indicators if term in query.lower())
        complexity += min(tech_count / 5.0, 0.3)
        
        # Question complexity
        question_words = ['what', 'how', 'when', 'where', 'why', 'which']
        if any(q in query.lower() for q in question_words):
            complexity += 0.2
        
        # Specific references (forms, numbers, etc.)
        import re
        if re.search(r'\b\d{3,4}\b', query):  # Form numbers like 2888
            complexity += 0.2
        
        return min(complexity, 1.0)
    
    def _analyze_result_quality(self, documents: List[RRFDocument]) -> float:
        """Analyze quality of RRF-scored results (0.0 to 1.0)."""
        if not documents:
            return 0.0
        
        # Score distribution analysis
        scores = [doc.rrf_score for doc in documents]
        score_range = max(scores) - min(scores)
        
        # Good quality if there's clear score separation
        quality = score_range
        
        # Bonus for high top scores
        if scores[0] > 0.8:  # Top document has high RRF score
            quality += 0.2
        
        # Penalty for very similar scores (less discriminative)
        if score_range < 0.1:
            quality *= 0.5
        
        return min(quality, 1.0)
    
    def _calculate_diversity_score(self, documents: List[RRFDocument]) -> float:
        """Calculate content diversity score (0.0 to 1.0)."""
        if len(documents) < 2:
            return 0.0
        
        # Simple diversity based on content length variation
        content_lengths = [len(doc.document.page_content) for doc in documents]
        if not content_lengths:
            return 0.0
        
        avg_length = sum(content_lengths) / len(content_lengths)
        variance = sum((length - avg_length) ** 2 for length in content_lengths) / len(content_lengths)
        
        # Normalize variance to 0-1 scale
        diversity = min(variance / (avg_length ** 2), 1.0) if avg_length > 0 else 0.0
        
        return diversity
    
    def _estimate_reranking_latency(self, doc_count: int) -> float:
        """Estimate reranking latency in milliseconds."""
        # Base latency + per-document cost
        base_latency = 100.0  # Base reranker initialization
        per_doc_latency = 50.0  # Per document processing
        
        return base_latency + (doc_count * per_doc_latency)
    
    def _calculate_reranking_confidence(
        self,
        query_complexity: float,
        uncertainty_score: float,
        result_quality: float,
        diversity_score: float
    ) -> float:
        """Calculate overall confidence in reranking benefits."""
        
        # Higher confidence for complex, uncertain queries
        complexity_factor = query_complexity * 0.3
        uncertainty_factor = uncertainty_score * 0.4
        
        # Lower confidence if results are already high quality
        quality_penalty = (1.0 - result_quality) * 0.2
        
        # Higher confidence with diverse results (more to optimize)
        diversity_bonus = diversity_score * 0.1
        
        confidence = complexity_factor + uncertainty_factor + quality_penalty + diversity_bonus
        return min(max(confidence, 0.0), 1.0)
    
    async def _execute_skip_strategy(
        self,
        documents: List[RRFDocument],
        decision: RerankingDecision,
        max_results: Optional[int] = None
    ) -> RerankingResult:
        """Execute skip strategy - return documents in RRF order."""
        final_docs = [Document(page_content=doc.document.page_content, metadata=doc.document.metadata) 
                     for doc in documents]
        
        if max_results:
            final_docs = final_docs[:max_results]
        
        return RerankingResult(
            documents=final_docs,
            strategy_used=RerankingStrategy.SKIP,
            reranker_confidence=0.0,
            reranking_time_ms=0.0
        )
    
    async def _execute_apply_strategy(
        self,
        query: str,
        documents: List[RRFDocument],
        decision: RerankingDecision,
        max_results: Optional[int] = None
    ) -> RerankingResult:
        """Execute apply strategy - use pure reranker scores."""
        try:
            # Convert to Documents for reranker
            doc_list = [Document(page_content=doc.document.page_content, metadata=doc.document.metadata) 
                       for doc in documents]
            
            # Apply reranking with timeout
            reranked_docs = await asyncio.wait_for(
                self.reranker_function(query, doc_list),
                timeout=self.max_reranking_latency_ms / 1000.0
            )
            
            if max_results:
                reranked_docs = reranked_docs[:max_results]
            
            # Calculate ranking changes
            ranking_changes = self._calculate_ranking_changes(documents, reranked_docs)
            
            return RerankingResult(
                documents=reranked_docs,
                strategy_used=RerankingStrategy.APPLY,
                reranker_confidence=decision.confidence,
                reranking_time_ms=0.0,  # Will be set by caller
                ranking_changes=ranking_changes
            )
            
        except Exception as e:
            logger.warning(f"Reranker failed, falling back to RRF scores: {e}")
            return await self._execute_fallback_strategy(documents, decision, max_results, str(e))
    
    async def _execute_blend_strategy(
        self,
        query: str,
        documents: List[RRFDocument],
        decision: RerankingDecision,
        max_results: Optional[int] = None
    ) -> RerankingResult:
        """Execute blend strategy - combine RRF and reranker scores."""
        try:
            # Get reranked documents
            doc_list = [Document(page_content=doc.document.page_content, metadata=doc.document.metadata) 
                       for doc in documents]
            
            reranked_docs = await asyncio.wait_for(
                self.reranker_function(query, doc_list),
                timeout=self.max_reranking_latency_ms / 1000.0
            )
            
            # Calculate blend ratio based on confidence
            rrf_weight = 1.0 - decision.confidence
            reranker_weight = decision.confidence
            
            # Blend scores and re-sort
            blended_docs = self._blend_document_scores(documents, reranked_docs, rrf_weight, reranker_weight)
            
            if max_results:
                blended_docs = blended_docs[:max_results]
            
            ranking_changes = self._calculate_ranking_changes(documents, blended_docs)
            
            return RerankingResult(
                documents=blended_docs,
                strategy_used=RerankingStrategy.BLEND,
                reranker_confidence=decision.confidence,
                blend_ratio=rrf_weight,
                reranking_time_ms=0.0,  # Will be set by caller
                ranking_changes=ranking_changes
            )
            
        except Exception as e:
            logger.warning(f"Blending failed, falling back to RRF scores: {e}")
            return await self._execute_fallback_strategy(documents, decision, max_results, str(e))
    
    async def _execute_fallback_strategy(
        self,
        documents: List[RRFDocument],
        decision: RerankingDecision,
        max_results: Optional[int] = None,
        fallback_reason: Optional[str] = None
    ) -> RerankingResult:
        """Execute fallback strategy - return to RRF scores due to failure."""
        final_docs = [Document(page_content=doc.document.page_content, metadata=doc.document.metadata) 
                     for doc in documents]
        
        if max_results:
            final_docs = final_docs[:max_results]
        
        return RerankingResult(
            documents=final_docs,
            strategy_used=RerankingStrategy.FALLBACK,
            reranker_confidence=0.0,
            reranking_time_ms=0.0,
            fallback_triggered=True,
            fallback_reason=fallback_reason or "Strategy fallback"
        )
    
    def _blend_document_scores(
        self,
        rrf_documents: List[RRFDocument],
        reranked_documents: List[Document],
        rrf_weight: float,
        reranker_weight: float
    ) -> List[Document]:
        """Blend RRF and reranker scores to create final ranking."""
        
        # Create mapping from content to RRF scores
        rrf_scores = {doc.document.page_content: doc.rrf_score for doc in rrf_documents}
        
        # Assign reranker scores based on position (higher position = higher score)
        reranker_scores = {}
        for i, doc in enumerate(reranked_documents):
            # Convert position to score (1.0 for first, decreasing)
            position_score = 1.0 - (i / len(reranked_documents))
            reranker_scores[doc.page_content] = position_score
        
        # Calculate blended scores
        blended_scores = []
        for doc in reranked_documents:
            rrf_score = rrf_scores.get(doc.page_content, 0.0)
            reranker_score = reranker_scores.get(doc.page_content, 0.0)
            
            blended_score = (rrf_weight * rrf_score) + (reranker_weight * reranker_score)
            blended_scores.append((doc, blended_score))
        
        # Sort by blended score
        blended_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, score in blended_scores]
    
    def _calculate_ranking_changes(
        self,
        original_docs: List[RRFDocument],
        final_docs: List[Document]
    ) -> int:
        """Calculate number of ranking position changes."""
        if len(original_docs) != len(final_docs):
            return len(final_docs)  # All positions changed
        
        # Create position mappings
        original_positions = {doc.document.page_content: i for i, doc in enumerate(original_docs)}
        
        changes = 0
        for i, doc in enumerate(final_docs):
            original_pos = original_positions.get(doc.page_content, -1)
            if original_pos != i:
                changes += 1
        
        return changes
    
    def _update_running_stats(self, result: RerankingResult):
        """Update running statistics."""
        if self.stats.total_requests == 0:
            self.stats.average_reranking_time_ms = result.reranking_time_ms
            self.stats.average_confidence = result.reranker_confidence
        else:
            # Running average
            alpha = 1.0 / self.stats.total_requests
            self.stats.average_reranking_time_ms = (
                (1 - alpha) * self.stats.average_reranking_time_ms + 
                alpha * result.reranking_time_ms
            )
            self.stats.average_confidence = (
                (1 - alpha) * self.stats.average_confidence + 
                alpha * result.reranker_confidence
            )
        
        # Update fallback rate
        fallback_count = self.stats.strategy_counts.get("fallback", 0)
        self.stats.fallback_rate = fallback_count / max(1, self.stats.total_requests)
    
    def get_stats(self) -> ConditionalRerankerStats:
        """Get current reranking statistics."""
        return self.stats
    
    def reset_stats(self):
        """Reset statistics tracking."""
        self.stats = ConditionalRerankerStats()


def create_conditional_reranker(
    reranker_function: Optional[callable] = None,
    **kwargs
) -> ConditionalReranker:
    """
    Factory function to create a conditional reranker with sensible defaults.
    
    Args:
        reranker_function: Async reranking function
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured ConditionalReranker instance
    """
    return ConditionalReranker(
        reranker_function=reranker_function,
        **kwargs
    )