"""
Adaptive K-selector for dynamic retrieval size adjustment.

This component analyzes query characteristics and retrieval confidence to
determine the optimal number of documents (K) to retrieve from each source.

K-selection strategy:
- Base K on query complexity and uncertainty
- Use coverage caps to prevent excessive retrieval
- Consider performance constraints and diminishing returns
- Adapt based on retriever-specific characteristics
"""

import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from app.components.uncertainty_scorer import UncertaintyScorer, UncertaintyResult
from app.components.bm25_gating import BM25Gate, BM25GatingResult
from app.core.logging import get_logger

logger = get_logger(__name__)


class QueryComplexity(Enum):
    """Query complexity classification."""
    SIMPLE = "simple"        # Single concept, clear intent
    MODERATE = "moderate"    # Multiple concepts, some ambiguity
    COMPLEX = "complex"      # Many concepts, high ambiguity
    EXPERT = "expert"        # Technical, specific, detailed


@dataclass
class KSelectionProfile:
    """K-selection parameters for different retriever types."""
    dense_k: int          # Number of docs from dense/semantic retrieval
    sparse_k: int         # Number of docs from sparse/lexical retrieval
    bm25_k: int          # Number of docs from BM25 (if activated)
    hybrid_k: int        # Number of docs from hybrid methods (if activated)
    
    @property
    def total_k(self) -> int:
        """Total documents across all retrievers."""
        return self.dense_k + self.sparse_k + self.bm25_k + self.hybrid_k
    
    @property
    def active_retrievers(self) -> List[str]:
        """List of retrievers with K > 0."""
        active = []
        if self.dense_k > 0:
            active.append("dense")
        if self.sparse_k > 0:
            active.append("sparse")
        if self.bm25_k > 0:
            active.append("bm25")
        if self.hybrid_k > 0:
            active.append("hybrid")
        return active


@dataclass
class KSelectionReasoning:
    """Detailed reasoning for K-selection decisions."""
    base_factors: Dict[str, float]        # Base factors influencing K
    complexity_multiplier: float         # Complexity-based adjustment
    uncertainty_boost: float             # Uncertainty-based boost
    coverage_caps_applied: bool          # Whether caps were applied
    performance_constraints: List[str]   # Performance limitations
    diminishing_returns_threshold: int   # Point where returns diminish
    
    def to_explanation(self) -> List[str]:
        """Convert to human-readable explanation."""
        explanations = []
        
        # Base factors
        if self.base_factors:
            factor_strs = [f"{k}={v:.2f}" for k, v in self.base_factors.items()]
            explanations.append(f"Base factors: {', '.join(factor_strs)}")
        
        # Adjustments
        if self.complexity_multiplier != 1.0:
            explanations.append(f"Complexity adjustment: {self.complexity_multiplier:.2f}x")
        
        if self.uncertainty_boost > 0:
            explanations.append(f"Uncertainty boost: +{self.uncertainty_boost:.1f}")
        
        # Constraints
        if self.coverage_caps_applied:
            explanations.append("Coverage caps applied to prevent over-retrieval")
        
        if self.performance_constraints:
            explanations.append(f"Performance constraints: {', '.join(self.performance_constraints)}")
        
        if self.diminishing_returns_threshold > 0:
            explanations.append(f"Diminishing returns beyond {self.diminishing_returns_threshold} docs")
        
        return explanations


@dataclass
class AdaptiveKResult:
    """Complete K-selection result with reasoning."""
    k_profile: KSelectionProfile
    query_complexity: QueryComplexity
    reasoning: KSelectionReasoning
    confidence: float                     # Confidence in K selection [0.0, 1.0]
    estimated_recall_coverage: float     # Expected recall coverage [0.0, 1.0]
    estimated_latency_ms: float          # Expected retrieval latency
    
    @property
    def retriever_ks(self) -> Dict[str, int]:
        """Get K values as dictionary for easy access."""
        return {
            "dense": self.k_profile.dense_k,
            "sparse": self.k_profile.sparse_k,
            "bm25": self.k_profile.bm25_k,
            "hybrid": self.k_profile.hybrid_k
        }


class AdaptiveKSelector:
    """
    Adaptive K-selector for dynamic retrieval size optimization.
    
    Analyzes query characteristics to determine optimal number of documents
    to retrieve from each available retriever, balancing recall and performance.
    """
    
    def __init__(
        self,
        uncertainty_scorer: Optional[UncertaintyScorer] = None,
        bm25_gate: Optional[BM25Gate] = None,
        base_k: int = 10,
        max_total_k: int = 50,
        performance_budget_ms: float = 200.0,
        enable_coverage_caps: bool = True,
        conservative_mode: bool = True
    ):
        """
        Initialize adaptive K-selector.
        
        Args:
            uncertainty_scorer: Query uncertainty analyzer
            bm25_gate: BM25 activation gate
            base_k: Base number of documents per retriever
            max_total_k: Maximum total documents across all retrievers
            performance_budget_ms: Maximum acceptable retrieval latency
            enable_coverage_caps: Whether to apply coverage caps
            conservative_mode: Prefer recall over speed
        """
        self.uncertainty_scorer = uncertainty_scorer or UncertaintyScorer()
        self.bm25_gate = bm25_gate or BM25Gate()
        self.base_k = base_k
        self.max_total_k = max_total_k
        self.performance_budget_ms = performance_budget_ms
        self.enable_coverage_caps = enable_coverage_caps
        self.conservative_mode = conservative_mode
        
        # Retriever-specific performance characteristics
        self.retriever_latency = {
            "dense": 45.0,      # ms per 10 documents
            "sparse": 25.0,     # ms per 10 documents  
            "bm25": 15.0,       # ms per 10 documents
            "hybrid": 60.0      # ms per 10 documents
        }
        
        # Complexity-based K multipliers
        self.complexity_multipliers = {
            QueryComplexity.SIMPLE: 0.8,     # Reduce K for simple queries
            QueryComplexity.MODERATE: 1.0,   # Standard K
            QueryComplexity.COMPLEX: 1.3,    # Increase K for complex queries
            QueryComplexity.EXPERT: 1.5      # Maximum K for expert queries
        }
        
        logger.info(f"AdaptiveKSelector initialized: base_k={base_k}, "
                   f"max_total_k={max_total_k}, budget={performance_budget_ms}ms")
    
    def select_k(self, query: str) -> AdaptiveKResult:
        """
        Select optimal K values for the given query.
        
        Args:
            query: Search query to analyze
            
        Returns:
            Complete K-selection result with reasoning
        """
        if not query or not query.strip():
            return self._create_minimal_result(query)
        
        query = query.strip()
        
        # Get uncertainty analysis
        uncertainty_result = self.uncertainty_scorer.score_query(query)
        
        # Get BM25 gating decision
        bm25_result = self.bm25_gate.should_activate_bm25(query)
        
        # Classify query complexity
        complexity = self._classify_complexity(query, uncertainty_result)
        
        # Calculate base K factors
        base_factors = self._calculate_base_factors(
            query, uncertainty_result, bm25_result, complexity
        )
        
        # Apply complexity multiplier
        complexity_multiplier = self.complexity_multipliers[complexity]
        
        # Calculate uncertainty boost
        uncertainty_boost = self._calculate_uncertainty_boost(uncertainty_result)
        
        # Determine initial K values
        initial_ks = self._calculate_initial_ks(
            base_factors, complexity_multiplier, uncertainty_boost, bm25_result
        )
        
        # Apply coverage caps and performance constraints
        final_ks, caps_applied, constraints = self._apply_constraints(
            initial_ks, complexity, uncertainty_result
        )
        
        # Calculate diminishing returns threshold
        diminishing_threshold = self._calculate_diminishing_returns_threshold(complexity)
        
        # Create reasoning
        reasoning = KSelectionReasoning(
            base_factors=base_factors,
            complexity_multiplier=complexity_multiplier,
            uncertainty_boost=uncertainty_boost,
            coverage_caps_applied=caps_applied,
            performance_constraints=constraints,
            diminishing_returns_threshold=diminishing_threshold
        )
        
        # Estimate performance characteristics
        confidence = self._estimate_selection_confidence(
            final_ks, uncertainty_result, complexity
        )
        
        recall_coverage = self._estimate_recall_coverage(
            final_ks, uncertainty_result, complexity
        )
        
        estimated_latency = self._estimate_retrieval_latency(final_ks)
        
        # Create final result
        k_profile = KSelectionProfile(
            dense_k=final_ks["dense"],
            sparse_k=final_ks["sparse"],
            bm25_k=final_ks["bm25"],
            hybrid_k=final_ks["hybrid"]
        )
        
        result = AdaptiveKResult(
            k_profile=k_profile,
            query_complexity=complexity,
            reasoning=reasoning,
            confidence=confidence,
            estimated_recall_coverage=recall_coverage,
            estimated_latency_ms=estimated_latency
        )
        
        logger.debug(f"K-selection for '{query[:50]}...': "
                    f"dense={final_ks['dense']}, sparse={final_ks['sparse']}, "
                    f"bm25={final_ks['bm25']}, total={k_profile.total_k}")
        
        return result
    
    def _classify_complexity(
        self, 
        query: str, 
        uncertainty_result: UncertaintyResult
    ) -> QueryComplexity:
        """Classify query complexity based on various indicators."""
        tokens = query.lower().split()
        
        # Simple indicators
        if len(tokens) <= 2:
            return QueryComplexity.SIMPLE
        
        # Expert indicators
        technical_terms = sum(1 for token in tokens if any(
            tech in token for tech in ["dnd", "cf", "form", "section", "paragraph", "regulation"]
        ))
        has_numbers = any(char.isdigit() for char in query)
        has_quotes = '"' in query or "'" in query
        
        # Check for specific technical/procedural terms
        procedural_terms = sum(1 for token in tokens if token in [
            "documentation", "requirements", "authorization", "approval", "procedures",
            "completion", "instructions", "directive", "policy", "administration"
        ])
        
        if technical_terms >= 2 or (technical_terms >= 1 and has_numbers):
            return QueryComplexity.EXPERT
        elif technical_terms >= 1 or procedural_terms >= 2:
            return QueryComplexity.COMPLEX
        
        # Complex indicators
        complex_indicators = 0
        
        if len(tokens) > 8:
            complex_indicators += 1
        
        if uncertainty_result.overall_uncertainty > 0.6:
            complex_indicators += 1
        
        boolean_terms = sum(1 for token in tokens if token in 
                           ["and", "or", "but", "not", "without", "except"])
        if boolean_terms > 1:
            complex_indicators += 1
        
        question_words = sum(1 for token in tokens if token in 
                           ["what", "how", "when", "where", "why", "which"])
        if question_words > 0 and len(tokens) > 5:
            complex_indicators += 1
        
        if complex_indicators >= 2:
            return QueryComplexity.COMPLEX
        elif complex_indicators >= 1 or len(tokens) > 5:
            return QueryComplexity.MODERATE
        else:
            return QueryComplexity.SIMPLE
    
    def _calculate_base_factors(
        self,
        query: str,
        uncertainty_result: UncertaintyResult,
        bm25_result: BM25GatingResult,
        complexity: QueryComplexity
    ) -> Dict[str, float]:
        """Calculate base factors influencing K selection."""
        factors = {}
        
        # Query length factor
        token_count = len(query.split())
        if token_count <= 3:
            factors["length"] = 0.8  # Shorter queries need fewer docs
        elif token_count >= 8:
            factors["length"] = 1.2  # Longer queries need more docs
        else:
            factors["length"] = 1.0
        
        # Uncertainty factor
        if uncertainty_result.overall_uncertainty > 0.7:
            factors["uncertainty"] = 1.3  # High uncertainty needs more docs
        elif uncertainty_result.overall_uncertainty < 0.3:
            factors["uncertainty"] = 0.9  # Low uncertainty needs fewer docs
        else:
            factors["uncertainty"] = 1.0
        
        # Domain coverage factor
        if uncertainty_result.features.coverage_score > 0.8:
            factors["coverage"] = 1.2  # Poor coverage needs more docs
        elif uncertainty_result.features.coverage_score < 0.2:
            factors["coverage"] = 0.9  # Good coverage needs fewer docs
        else:
            factors["coverage"] = 1.0
        
        # Specificity factor
        specificity_raw = 1.0 - uncertainty_result.features.specificity_score
        if specificity_raw > 0.8:
            factors["specificity"] = 0.8  # Highly specific queries need fewer docs
        elif specificity_raw < 0.3:
            factors["specificity"] = 1.2  # Vague queries need more docs
        else:
            factors["specificity"] = 1.0
        
        return factors
    
    def _calculate_uncertainty_boost(self, uncertainty_result: UncertaintyResult) -> float:
        """Calculate additional K boost based on uncertainty level."""
        if uncertainty_result.confidence_level == "low":
            return 0.3  # Boost K by 30% for low confidence
        elif uncertainty_result.confidence_level == "medium":
            return 0.1  # Small boost for medium confidence
        else:
            return 0.0  # No boost for high confidence
    
    def _calculate_initial_ks(
        self,
        base_factors: Dict[str, float],
        complexity_multiplier: float,
        uncertainty_boost: float,
        bm25_result: BM25GatingResult
    ) -> Dict[str, int]:
        """Calculate initial K values before applying constraints."""
        
        # Calculate base K from factors
        factor_product = 1.0
        for factor_value in base_factors.values():
            factor_product *= factor_value
        
        base_adjusted_k = self.base_k * factor_product * complexity_multiplier
        uncertainty_adjusted_k = base_adjusted_k * (1.0 + uncertainty_boost)
        
        # Distribute K across retrievers
        ks = {
            "dense": 0,
            "sparse": 0, 
            "bm25": 0,
            "hybrid": 0
        }
        
        # Always allocate to dense (semantic) retrieval
        ks["dense"] = max(5, int(uncertainty_adjusted_k * 0.6))  # 60% to dense
        
        # Always allocate to sparse retrieval  
        ks["sparse"] = max(3, int(uncertainty_adjusted_k * 0.3))  # 30% to sparse
        
        # Conditionally allocate to BM25
        if bm25_result.should_activate:
            ks["bm25"] = max(2, int(uncertainty_adjusted_k * 0.25))  # 25% to BM25
        
        # Conditionally allocate to hybrid (for very complex queries)
        if uncertainty_boost > 0.2 and complexity_multiplier >= 1.3:
            ks["hybrid"] = max(2, int(uncertainty_adjusted_k * 0.15))  # 15% to hybrid
        
        return ks
    
    def _apply_constraints(
        self,
        initial_ks: Dict[str, int],
        complexity: QueryComplexity,
        uncertainty_result: UncertaintyResult
    ) -> Tuple[Dict[str, int], bool, List[str]]:
        """Apply coverage caps and performance constraints."""
        constrained_ks = initial_ks.copy()
        caps_applied = False
        constraints = []
        
        # Apply total K cap
        initial_total = sum(initial_ks.values())
        if initial_total > self.max_total_k:
            # Proportionally reduce all K values
            scale_factor = self.max_total_k / initial_total
            for retriever in constrained_ks:
                if constrained_ks[retriever] > 0:
                    constrained_ks[retriever] = max(1, int(constrained_ks[retriever] * scale_factor))
            caps_applied = True
            constraints.append(f"Total K capped at {self.max_total_k}")
        
        # Apply performance constraints
        estimated_latency = self._estimate_retrieval_latency(constrained_ks)
        if estimated_latency > self.performance_budget_ms:
            # Reduce K values to meet performance budget
            reduction_factor = self.performance_budget_ms / estimated_latency
            for retriever in constrained_ks:
                if constrained_ks[retriever] > 0:
                    constrained_ks[retriever] = max(1, int(constrained_ks[retriever] * reduction_factor))
            constraints.append(f"Performance budget constraint: {self.performance_budget_ms}ms")
        
        # Apply coverage caps if enabled
        if self.enable_coverage_caps:
            # Cap individual retrievers to prevent over-reliance
            max_individual_k = min(25, self.max_total_k // 2)
            
            for retriever in constrained_ks:
                if constrained_ks[retriever] > max_individual_k:
                    constrained_ks[retriever] = max_individual_k
                    caps_applied = True
        
        # Ensure minimum viable K values (but respect total cap)
        current_total = sum(constrained_ks.values())
        if current_total < self.max_total_k:
            # Only apply minimums if we have room
            if constrained_ks["dense"] > 0 and constrained_ks["dense"] < 3:
                constrained_ks["dense"] = min(3, self.max_total_k - (current_total - constrained_ks["dense"]))
            if constrained_ks["sparse"] > 0 and constrained_ks["sparse"] < 2:
                current_total = sum(constrained_ks.values())
                if current_total < self.max_total_k:
                    constrained_ks["sparse"] = min(2, self.max_total_k - (current_total - constrained_ks["sparse"]))
        
        # Final check to ensure we don't exceed total cap
        final_total = sum(constrained_ks.values())
        if final_total > self.max_total_k:
            # Emergency reduction - remove from least important retrievers first
            excess = final_total - self.max_total_k
            priority = ["hybrid", "bm25", "sparse", "dense"]  # Remove in this order
            
            for retriever in priority:
                if excess <= 0:
                    break
                reduction = min(constrained_ks[retriever], excess)
                constrained_ks[retriever] -= reduction
                excess -= reduction
                if constrained_ks[retriever] == 0 and excess > 0:
                    continue
        
        return constrained_ks, caps_applied, constraints
    
    def _calculate_diminishing_returns_threshold(self, complexity: QueryComplexity) -> int:
        """Calculate the point where additional documents provide diminishing returns."""
        base_thresholds = {
            QueryComplexity.SIMPLE: 8,
            QueryComplexity.MODERATE: 15,
            QueryComplexity.COMPLEX: 25,
            QueryComplexity.EXPERT: 35
        }
        
        threshold = base_thresholds[complexity]
        
        if self.conservative_mode:
            threshold = int(threshold * 1.2)  # More conservative = higher threshold
        
        return threshold
    
    def _estimate_selection_confidence(
        self,
        final_ks: Dict[str, int],
        uncertainty_result: UncertaintyResult,
        complexity: QueryComplexity
    ) -> float:
        """Estimate confidence in the K selection."""
        confidence_factors = []
        
        # Base confidence from uncertainty analysis
        base_confidence = 1.0 - uncertainty_result.overall_uncertainty
        confidence_factors.append(base_confidence)
        
        # Complexity appropriateness
        total_k = sum(final_ks.values())
        expected_k_ranges = {
            QueryComplexity.SIMPLE: (5, 15),
            QueryComplexity.MODERATE: (10, 25),
            QueryComplexity.COMPLEX: (15, 35),
            QueryComplexity.EXPERT: (20, 45)
        }
        
        min_expected, max_expected = expected_k_ranges[complexity]
        if min_expected <= total_k <= max_expected:
            confidence_factors.append(0.9)  # Good K range
        else:
            confidence_factors.append(0.6)  # Suboptimal K range
        
        # Retriever diversity
        active_retrievers = sum(1 for k in final_ks.values() if k > 0)
        if active_retrievers >= 2:
            confidence_factors.append(0.8)  # Good diversity
        else:
            confidence_factors.append(0.5)  # Limited diversity
        
        # Return average confidence
        return sum(confidence_factors) / len(confidence_factors)
    
    def _estimate_recall_coverage(
        self,
        final_ks: Dict[str, int],
        uncertainty_result: UncertaintyResult,
        complexity: QueryComplexity
    ) -> float:
        """Estimate expected recall coverage."""
        
        # Base coverage from total K
        total_k = sum(final_ks.values())
        base_coverage = min(0.95, 0.3 + (total_k / 50) * 0.6)  # Asymptotic approach to 95%
        
        # Adjust for query characteristics
        adjustments = []
        
        # Domain coverage adjustment
        if uncertainty_result.features.coverage_score > 0.8:
            adjustments.append(-0.15)  # Out-of-domain reduces expected coverage
        elif uncertainty_result.features.coverage_score < 0.2:
            adjustments.append(0.1)   # In-domain improves coverage
        
        # Complexity adjustment
        complexity_adjustments = {
            QueryComplexity.SIMPLE: 0.05,    # Simple queries easier to cover
            QueryComplexity.MODERATE: 0.0,   # Baseline
            QueryComplexity.COMPLEX: -0.05,  # Complex queries harder to cover
            QueryComplexity.EXPERT: -0.1     # Expert queries most challenging
        }
        adjustments.append(complexity_adjustments[complexity])
        
        # Retriever diversity bonus
        active_retrievers = sum(1 for k in final_ks.values() if k > 0)
        if active_retrievers >= 3:
            adjustments.append(0.1)  # Diversity bonus
        
        # Apply adjustments
        final_coverage = base_coverage + sum(adjustments)
        return max(0.1, min(0.95, final_coverage))  # Clamp to reasonable range
    
    def _estimate_retrieval_latency(self, ks: Dict[str, int]) -> float:
        """Estimate total retrieval latency in milliseconds."""
        total_latency = 0.0
        
        for retriever, k in ks.items():
            if k > 0:
                # Base latency per 10 documents
                base_latency = self.retriever_latency[retriever]
                # Scale by actual K
                retriever_latency = base_latency * (k / 10.0)
                total_latency += retriever_latency
        
        # Add coordination overhead (parallel execution assumed)
        coordination_overhead = min(20.0, total_latency * 0.1)
        
        return total_latency + coordination_overhead
    
    def _create_minimal_result(self, query: str) -> AdaptiveKResult:
        """Create minimal result for empty/invalid queries."""
        minimal_ks = KSelectionProfile(
            dense_k=3,    # Minimal dense retrieval
            sparse_k=2,   # Minimal sparse retrieval
            bm25_k=0,     # No BM25 for empty queries
            hybrid_k=0    # No hybrid for empty queries
        )
        
        reasoning = KSelectionReasoning(
            base_factors={"empty_query": 0.0},
            complexity_multiplier=0.5,
            uncertainty_boost=0.0,
            coverage_caps_applied=False,
            performance_constraints=["Minimal retrieval for empty query"],
            diminishing_returns_threshold=5
        )
        
        return AdaptiveKResult(
            k_profile=minimal_ks,
            query_complexity=QueryComplexity.SIMPLE,
            reasoning=reasoning,
            confidence=0.3,
            estimated_recall_coverage=0.2,
            estimated_latency_ms=30.0
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics and configuration."""
        return {
            "config": {
                "base_k": self.base_k,
                "max_total_k": self.max_total_k,
                "performance_budget_ms": self.performance_budget_ms,
                "enable_coverage_caps": self.enable_coverage_caps,
                "conservative_mode": self.conservative_mode
            },
            "retriever_latency": self.retriever_latency,
            "complexity_multipliers": {k.value: v for k, v in self.complexity_multipliers.items()}
        }
    
    def update_performance_budget(self, new_budget_ms: float) -> None:
        """Update the performance budget."""
        old_budget = self.performance_budget_ms
        self.performance_budget_ms = new_budget_ms
        logger.info(f"Performance budget updated: {old_budget}ms -> {new_budget_ms}ms")


def create_adaptive_k_selector(
    base_k: int = 10,
    max_total_k: int = 50,
    performance_budget_ms: float = 200.0,
    conservative_mode: bool = True,
    uncertainty_scorer: Optional[UncertaintyScorer] = None,
    bm25_gate: Optional[BM25Gate] = None
) -> AdaptiveKSelector:
    """
    Factory function to create adaptive K-selector with recommended settings.
    
    Args:
        base_k: Base number of documents per retriever
        max_total_k: Maximum total documents across all retrievers  
        performance_budget_ms: Maximum acceptable retrieval latency
        conservative_mode: Prefer recall over speed
        uncertainty_scorer: Custom uncertainty scorer
        bm25_gate: Custom BM25 gate
        
    Returns:
        Configured AdaptiveKSelector instance
    """
    return AdaptiveKSelector(
        uncertainty_scorer=uncertainty_scorer,
        bm25_gate=bm25_gate,
        base_k=base_k,
        max_total_k=max_total_k,
        performance_budget_ms=performance_budget_ms,
        enable_coverage_caps=True,
        conservative_mode=conservative_mode
    )