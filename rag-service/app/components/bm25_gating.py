"""
BM25 Smart Gating Logic for adaptive keyword retrieval activation.

This component analyzes query characteristics and uncertainty scores to determine
when BM25 keyword retrieval should be activated alongside semantic retrieval.

Gating strategy:
- Always use dense (semantic) retrieval as the foundation
- Activate BM25 when keyword matching would likely improve results
- Consider query complexity, uncertainty, and domain coverage
- Use performance budgets to limit expensive operations
"""

import time
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from enum import Enum
import re

from app.components.uncertainty_scorer import UncertaintyScorer, UncertaintyResult
from app.core.logging import get_logger

logger = get_logger(__name__)


class GatingDecision(Enum):
    """BM25 gating decision types."""
    SKIP = "skip"           # Skip BM25, use dense only
    ACTIVATE = "activate"   # Activate BM25 alongside dense
    REQUIRED = "required"   # BM25 strongly recommended


@dataclass
class BM25GatingFeatures:
    """Features used for BM25 gating decisions."""
    has_exact_phrases: bool          # Query contains quoted phrases
    has_technical_terms: bool        # Contains specific technical vocabulary
    has_numbers_codes: bool          # Contains numbers, codes, identifiers
    has_negation: bool              # Contains NOT, without, except
    high_keyword_density: bool       # High ratio of important keywords
    out_of_domain: bool             # Query outside primary domain
    high_uncertainty: bool          # Uncertainty scorer indicates confusion
    complex_boolean: bool           # Contains AND/OR logic
    
    def __post_init__(self):
        """Validate feature consistency."""
        if self.has_exact_phrases and not self.has_technical_terms:
            # Exact phrases often indicate technical queries
            pass  # This is valid - could be exact but not technical


@dataclass
class BM25GatingResult:
    """Complete BM25 gating decision with reasoning."""
    decision: GatingDecision
    confidence: float               # Confidence in the decision [0.0, 1.0]
    features: BM25GatingFeatures
    reasoning: List[str]           # Human-readable explanations
    estimated_benefit: float       # Expected retrieval improvement [0.0, 1.0]
    performance_cost: float        # Estimated additional latency in ms
    
    @property
    def should_activate(self) -> bool:
        """Whether BM25 should be activated."""
        return self.decision in [GatingDecision.ACTIVATE, GatingDecision.REQUIRED]


@dataclass
class BM25GatingStats:
    """Statistics for BM25 gating decisions."""
    total_queries: int
    bm25_activated: int
    bm25_skipped: int
    avg_decision_time_ms: float
    activation_rate: float
    
    @property
    def skip_rate(self) -> float:
        """Rate of BM25 skipping decisions."""
        return 1.0 - self.activation_rate if self.activation_rate <= 1.0 else 0.0


class BM25Gate:
    """
    Smart gating system for BM25 keyword retrieval activation.
    
    Uses query analysis and uncertainty scoring to make intelligent decisions
    about when keyword-based retrieval would improve results.
    """
    
    # Keywords that strongly benefit from exact matching
    EXACT_MATCH_KEYWORDS = {
        'form', 'section', 'paragraph', 'article', 'clause', 'regulation',
        'policy', 'directive', 'dnd', 'cf', 'receipt', 'claim', 'authorization',
        'approval', 'maximum', 'minimum', 'rate', 'allowance', 'expense'
    }
    
    # Technical terms that benefit from keyword retrieval
    TECHNICAL_TERMS = {
        'accommodation', 'reimbursement', 'per diem', 'temporary duty', 'posting',
        'relocation', 'pcs', 'exercise', 'training', 'deployment', 'leave',
        'treasury board', 'financial administration', 'entitlement', 'conus',
        'oconus', 'tdv', 'tdy', 'documentation', 'criteria', 'procedure'
    }
    
    # Negation words that require precise keyword matching
    NEGATION_TERMS = {
        'not', 'no', 'never', 'without', 'except', 'excluding', 'unless',
        'neither', 'nor', 'cannot', 'dont', "don't", 'wont', "won't"
    }
    
    def __init__(
        self,
        uncertainty_scorer: Optional[UncertaintyScorer] = None,
        performance_budget_ms: float = 50.0,
        activation_threshold: float = 0.6,
        conservative_mode: bool = True
    ):
        """
        Initialize BM25 gating system.
        
        Args:
            uncertainty_scorer: Uncertainty scorer for query analysis
            performance_budget_ms: Maximum acceptable BM25 latency overhead
            activation_threshold: Threshold for activating BM25 [0.0, 1.0]
            conservative_mode: If True, prefer recall over speed
        """
        self.uncertainty_scorer = uncertainty_scorer or UncertaintyScorer()
        self.performance_budget_ms = performance_budget_ms
        self.activation_threshold = activation_threshold
        self.conservative_mode = conservative_mode
        
        # Statistics tracking
        self._stats = BM25GatingStats(
            total_queries=0,
            bm25_activated=0,
            bm25_skipped=0,
            avg_decision_time_ms=0.0,
            activation_rate=0.0
        )
        
        # Decision time tracking for running average
        self._decision_times: List[float] = []
        
        logger.info(f"BM25Gate initialized: budget={performance_budget_ms}ms, "
                   f"threshold={activation_threshold}, conservative={conservative_mode}")
    
    def should_activate_bm25(self, query: str) -> BM25GatingResult:
        """
        Determine whether BM25 should be activated for the given query.
        
        Args:
            query: Search query to analyze
            
        Returns:
            Complete gating decision with reasoning
        """
        start_time = time.time()
        
        try:
            # Update statistics
            self._stats.total_queries += 1
            
            # Handle edge cases
            if not query or not query.strip():
                result = self._create_skip_result(
                    features=self._extract_features(""),
                    reasoning=["Empty query - no BM25 benefit"]
                )
                # Update statistics for empty query
                self._stats.bm25_skipped += 1
                self._stats.activation_rate = self._stats.bm25_activated / self._stats.total_queries
                return result
            
            query = query.strip()
            
            # Extract gating features
            features = self._extract_features(query)
            
            # Get uncertainty analysis
            uncertainty_result = self.uncertainty_scorer.score_query(query)
            
            # Make gating decision
            decision, confidence, reasoning, benefit = self._make_decision(
                features, uncertainty_result, query
            )
            
            # Estimate performance cost
            performance_cost = self._estimate_performance_cost(query, features)
            
            # Create result
            result = BM25GatingResult(
                decision=decision,
                confidence=confidence,
                features=features,
                reasoning=reasoning,
                estimated_benefit=benefit,
                performance_cost=performance_cost
            )
            
            # Update statistics
            if result.should_activate:
                self._stats.bm25_activated += 1
            else:
                self._stats.bm25_skipped += 1
            
            self._stats.activation_rate = self._stats.bm25_activated / self._stats.total_queries
            
            # Track decision time
            decision_time_ms = (time.time() - start_time) * 1000
            self._decision_times.append(decision_time_ms)
            if len(self._decision_times) > 100:  # Keep last 100 measurements
                self._decision_times = self._decision_times[-100:]
            
            self._stats.avg_decision_time_ms = sum(self._decision_times) / len(self._decision_times)
            
            logger.debug(f"BM25 gating decision: {decision.value} (confidence: {confidence:.2f}) "
                        f"for query: '{query[:50]}...' in {decision_time_ms:.1f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in BM25 gating: {e}")
            # Fallback to conservative activation
            return self._create_activation_result(
                features=self._extract_features(query),
                reasoning=[f"Error in analysis, activating conservatively: {str(e)}"]
            )
    
    def _extract_features(self, query: str) -> BM25GatingFeatures:
        """Extract features relevant for BM25 gating decisions."""
        if not query:
            return BM25GatingFeatures(
                has_exact_phrases=False,
                has_technical_terms=False,
                has_numbers_codes=False,
                has_negation=False,
                high_keyword_density=False,
                out_of_domain=False,
                high_uncertainty=False,
                complex_boolean=False
            )
        
        query_lower = query.lower()
        tokens = query_lower.split()
        
        # Check for exact phrases (quoted text)
        has_exact_phrases = '"' in query or "'" in query
        
        # Check for technical terms (both single tokens and phrases)
        technical_count = sum(1 for token in tokens if token in self.TECHNICAL_TERMS)
        exact_match_count = sum(1 for token in tokens if token in self.EXACT_MATCH_KEYWORDS)
        
        # Also check for multi-word technical phrases
        phrase_count = sum(1 for phrase in self.TECHNICAL_TERMS if ' ' in phrase and phrase in query_lower)
        
        has_technical_terms = (technical_count > 0) or (exact_match_count > 0) or (phrase_count > 0)
        
        # Check for numbers and codes
        has_numbers_codes = bool(re.search(r'\d', query)) or bool(re.search(r'[A-Z]{2,}', query))
        
        # Check for negation
        has_negation = any(token in self.NEGATION_TERMS for token in tokens)
        
        # Check keyword density
        important_keywords = technical_count + exact_match_count
        keyword_density = important_keywords / max(1, len(tokens))
        high_keyword_density = keyword_density > 0.3
        
        # Domain coverage (will be updated with uncertainty result)
        out_of_domain = False  # Placeholder
        high_uncertainty = False  # Placeholder
        
        # Check for complex boolean logic
        boolean_terms = {'and', 'or', 'but', 'however', 'also', 'additionally', 'furthermore'}
        complex_boolean = sum(1 for token in tokens if token in boolean_terms) > 1
        
        return BM25GatingFeatures(
            has_exact_phrases=has_exact_phrases,
            has_technical_terms=has_technical_terms,
            has_numbers_codes=has_numbers_codes,
            has_negation=has_negation,
            high_keyword_density=high_keyword_density,
            out_of_domain=out_of_domain,
            high_uncertainty=high_uncertainty,
            complex_boolean=complex_boolean
        )
    
    def _make_decision(
        self,
        features: BM25GatingFeatures,
        uncertainty_result: UncertaintyResult,
        query: str
    ) -> Tuple[GatingDecision, float, List[str], float]:
        """
        Make the core BM25 gating decision.
        
        Returns:
            Tuple of (decision, confidence, reasoning, estimated_benefit)
        """
        # Update features with uncertainty information
        features.out_of_domain = uncertainty_result.features.coverage_score > 0.7
        features.high_uncertainty = uncertainty_result.overall_uncertainty > 0.6
        
        reasoning = []
        activation_factors = []
        
        # Analyze activation factors
        
        # 1. Exact phrases strongly favor BM25
        if features.has_exact_phrases:
            activation_factors.append(0.9)
            reasoning.append("Query contains exact phrases - BM25 excels at exact matching")
        
        # 2. Technical terms benefit from keyword matching
        if features.has_technical_terms:
            activation_factors.append(0.7)
            reasoning.append("Technical terms detected - keyword matching recommended")
        
        # 3. Numbers and codes need precise matching
        if features.has_numbers_codes:
            activation_factors.append(0.8)
            reasoning.append("Numbers/codes present - exact matching beneficial")
        
        # 4. Negation requires careful keyword handling
        if features.has_negation:
            activation_factors.append(0.6)
            reasoning.append("Negation detected - keyword precision important")
        
        # 5. High keyword density suggests structured query
        if features.high_keyword_density:
            activation_factors.append(0.5)
            reasoning.append("High keyword density - structured query benefits from BM25")
        
        # 6. Out-of-domain queries need broader retrieval
        if features.out_of_domain:
            activation_factors.append(0.6)
            reasoning.append("Out-of-domain query - keyword fallback recommended")
        
        # 7. High uncertainty suggests need for multiple approaches
        if features.high_uncertainty:
            activation_factors.append(0.5)
            reasoning.append("High query uncertainty - multiple retrieval methods beneficial")
        
        # 8. Complex boolean logic benefits from keyword handling
        if features.complex_boolean:
            activation_factors.append(0.4)
            reasoning.append("Complex boolean logic - BM25 handles structured queries well")
        
        # Calculate overall activation score
        if not activation_factors:
            activation_score = 0.2  # Default low activation for simple queries
            reasoning.append("Simple semantic query - dense retrieval likely sufficient")
        else:
            activation_score = max(activation_factors)  # Take strongest factor
        
        # Apply conservative bias if enabled
        if self.conservative_mode and activation_score > 0.3:
            activation_score = min(1.0, activation_score + 0.1)
            if len(reasoning) == len(activation_factors) + (1 if not activation_factors else 0):
                reasoning.append("Conservative mode - slightly favoring recall over speed")
        
        # Make decision based on threshold
        if activation_score >= self.activation_threshold:
            if activation_score >= 0.8:
                decision = GatingDecision.REQUIRED
            else:
                decision = GatingDecision.ACTIVATE
        else:
            decision = GatingDecision.SKIP
        
        # Calculate confidence (how certain we are about the decision)
        if decision == GatingDecision.SKIP:
            confidence = 1.0 - activation_score  # More confident when score is low
        else:
            confidence = activation_score  # More confident when score is high
        
        confidence = max(0.1, min(0.95, confidence))  # Keep in reasonable range
        
        # Estimate benefit (how much BM25 is expected to help)
        estimated_benefit = activation_score
        
        return decision, confidence, reasoning, estimated_benefit
    
    def _estimate_performance_cost(
        self, 
        query: str, 
        features: BM25GatingFeatures
    ) -> float:
        """
        Estimate the performance cost of activating BM25.
        
        Returns:
            Estimated additional latency in milliseconds
        """
        base_cost = 15.0  # Base BM25 query cost
        
        # Query length affects processing time
        query_length_factor = min(2.0, len(query.split()) / 10.0)
        length_cost = base_cost * query_length_factor
        
        # Complex features increase cost
        complexity_multiplier = 1.0
        if features.has_exact_phrases:
            complexity_multiplier += 0.2
        if features.complex_boolean:
            complexity_multiplier += 0.3
        if features.has_negation:
            complexity_multiplier += 0.1
        
        total_cost = length_cost * complexity_multiplier
        
        # Cap at reasonable maximum
        return min(self.performance_budget_ms, total_cost)
    
    def _create_skip_result(
        self, 
        features: BM25GatingFeatures, 
        reasoning: List[str],
        confidence: float = 0.8
    ) -> BM25GatingResult:
        """Create a SKIP decision result."""
        return BM25GatingResult(
            decision=GatingDecision.SKIP,
            confidence=confidence,
            features=features,
            reasoning=reasoning,
            estimated_benefit=0.1,
            performance_cost=0.0
        )
    
    def _create_activation_result(
        self, 
        features: BM25GatingFeatures, 
        reasoning: List[str],
        confidence: float = 0.7
    ) -> BM25GatingResult:
        """Create an ACTIVATE decision result."""
        return BM25GatingResult(
            decision=GatingDecision.ACTIVATE,
            confidence=confidence,
            features=features,
            reasoning=reasoning,
            estimated_benefit=0.6,
            performance_cost=20.0
        )
    
    def get_stats(self) -> BM25GatingStats:
        """Get current gating statistics."""
        return self._stats
    
    def reset_stats(self) -> None:
        """Reset gating statistics."""
        self._stats = BM25GatingStats(
            total_queries=0,
            bm25_activated=0,
            bm25_skipped=0,
            avg_decision_time_ms=0.0,
            activation_rate=0.0
        )
        self._decision_times = []
        logger.info("BM25Gate statistics reset")
    
    def update_performance_budget(self, new_budget_ms: float) -> None:
        """Update the performance budget for BM25 operations."""
        old_budget = self.performance_budget_ms
        self.performance_budget_ms = new_budget_ms
        logger.info(f"BM25Gate performance budget updated: {old_budget}ms -> {new_budget_ms}ms")
    
    def analyze_query_batch(
        self, 
        queries: List[str]
    ) -> Dict[str, BM25GatingResult]:
        """
        Analyze a batch of queries for testing and optimization.
        
        Args:
            queries: List of queries to analyze
            
        Returns:
            Dictionary mapping queries to gating results
        """
        results = {}
        
        start_time = time.time()
        for query in queries:
            results[query] = self.should_activate_bm25(query)
        
        batch_time = (time.time() - start_time) * 1000
        avg_time_per_query = batch_time / max(1, len(queries))
        
        logger.info(f"Analyzed {len(queries)} queries in {batch_time:.1f}ms "
                   f"(avg: {avg_time_per_query:.1f}ms/query)")
        
        return results


def create_bm25_gate(
    uncertainty_scorer: Optional[UncertaintyScorer] = None,
    performance_budget_ms: float = 50.0,
    activation_threshold: float = 0.6,
    conservative_mode: bool = True
) -> BM25Gate:
    """
    Factory function to create BM25 gate with recommended settings.
    
    Args:
        uncertainty_scorer: Custom uncertainty scorer
        performance_budget_ms: Maximum acceptable BM25 latency overhead
        activation_threshold: Threshold for activating BM25
        conservative_mode: Prefer recall over speed
        
    Returns:
        Configured BM25Gate instance
    """
    return BM25Gate(
        uncertainty_scorer=uncertainty_scorer,
        performance_budget_ms=performance_budget_ms,
        activation_threshold=activation_threshold,
        conservative_mode=conservative_mode
    )