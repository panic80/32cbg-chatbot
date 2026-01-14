"""
Multi-feature uncertainty scorer for query analysis.

This component analyzes query characteristics to determine retrieval confidence
and guide adaptive retrieval decisions (e.g., which retrievers to activate).

Uncertainty factors:
- Query ambiguity (polysemy, vague terms)
- Query specificity (length, detail level)
- Domain coverage (how well query matches known content)
- Semantic coherence (internal consistency)
"""

import re
import math
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
import string
from collections import Counter, defaultdict

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class UncertaintyFeatures:
    """Individual uncertainty feature scores."""
    ambiguity_score: float      # 0.0 = specific, 1.0 = ambiguous
    specificity_score: float    # 0.0 = vague, 1.0 = specific  
    coverage_score: float       # 0.0 = well-covered, 1.0 = out-of-domain
    coherence_score: float      # 0.0 = coherent, 1.0 = incoherent
    complexity_score: float     # 0.0 = simple, 1.0 = complex
    
    @property
    def feature_dict(self) -> Dict[str, float]:
        """Get features as dictionary for analysis."""
        return {
            'ambiguity': self.ambiguity_score,
            'specificity': self.specificity_score, 
            'coverage': self.coverage_score,
            'coherence': self.coherence_score,
            'complexity': self.complexity_score
        }


@dataclass  
class UncertaintyResult:
    """Complete uncertainty analysis result."""
    overall_uncertainty: float  # Final weighted uncertainty score [0.0, 1.0]
    confidence_level: str      # 'high', 'medium', 'low'
    features: UncertaintyFeatures
    reasoning: List[str]       # Human-readable explanations
    retriever_recommendations: Dict[str, bool]  # Which retrievers to activate


class UncertaintyScorer:
    """
    Multi-feature uncertainty scorer for retrieval queries.
    
    Analyzes query characteristics to determine retrieval confidence and
    provide recommendations for adaptive retrieval strategies.
    """
    
    # CF travel domain vocabulary for coverage analysis
    CF_TRAVEL_TERMS = {
        'travel', 'allowance', 'expense', 'reimbursement', 'per diem', 'accommodation',
        'lodging', 'meal', 'transportation', 'mileage', 'flight', 'hotel', 'receipt',
        'claim', 'duty', 'official', 'temporary', 'posting', 'relocation', 'pcs',
        'exercise', 'training', 'deployment', 'leave', 'canadian forces', 'cf',
        'military', 'personnel', 'member', 'rank', 'policy', 'directive', 'regulation',
        'treasury board', 'financial', 'administration', 'authorization', 'approval',
        'entitlement', 'rate', 'maximum', 'minimum', 'standard', 'exceptional',
        'domestic', 'international', 'overseas', 'conus', 'oconus', 'tdv', 'tdy'
    }
    
    # Ambiguous terms that could have multiple meanings
    AMBIGUOUS_TERMS = {
        'allowance', 'rate', 'standard', 'normal', 'regular', 'special', 'general',
        'basic', 'additional', 'extra', 'other', 'various', 'different', 'certain',
        'appropriate', 'reasonable', 'necessary', 'required', 'applicable', 'relevant',
        'information', 'details', 'about', 'regarding', 'concerning', 'stuff', 'things'
    }
    
    # High-specificity terms that indicate detailed queries
    SPECIFIC_TERMS = {
        'receipt', 'documentation', 'form', 'claim', 'application', 'approval',
        'authorization', 'calculation', 'formula', 'procedure', 'process', 'step',
        'requirement', 'criteria', 'condition', 'limitation', 'restriction', 'deadline'
    }
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        confidence_thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Initialize uncertainty scorer.
        
        Args:
            weights: Feature weights for final score calculation
            confidence_thresholds: Thresholds for confidence level classification
        """
        self.weights = weights or {
            'ambiguity': 0.25,
            'specificity': 0.20,
            'coverage': 0.25,
            'coherence': 0.15,
            'complexity': 0.15
        }
        
        self.confidence_thresholds = confidence_thresholds or {
            'high': 0.3,     # uncertainty <= 0.3 = high confidence
            'medium': 0.6,   # 0.3 < uncertainty <= 0.6 = medium confidence  
            'low': 1.0       # uncertainty > 0.6 = low confidence
        }
        
        # Validate weights sum to 1.0
        weight_sum = sum(self.weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            logger.warning(f"Feature weights sum to {weight_sum:.3f}, not 1.0")
    
    def score_query(self, query: str) -> UncertaintyResult:
        """
        Analyze query and return complete uncertainty assessment.
        
        Args:
            query: Search query to analyze
            
        Returns:
            Complete uncertainty analysis result
        """
        if query is None or not query or not query.strip():
            return self._create_empty_result()
        
        query = query.strip()
        
        # Extract features
        features = self._extract_features(query)
        
        # Calculate weighted overall uncertainty
        overall_uncertainty = (
            features.ambiguity_score * self.weights['ambiguity'] +
            features.specificity_score * self.weights['specificity'] +
            features.coverage_score * self.weights['coverage'] +
            features.coherence_score * self.weights['coherence'] +
            features.complexity_score * self.weights['complexity']
        )
        
        # Determine confidence level
        confidence_level = self._classify_confidence(overall_uncertainty)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(features, query)
        
        # Generate retriever recommendations
        retriever_recs = self._recommend_retrievers(overall_uncertainty, features)
        
        return UncertaintyResult(
            overall_uncertainty=overall_uncertainty,
            confidence_level=confidence_level,
            features=features,
            reasoning=reasoning,
            retriever_recommendations=retriever_recs
        )
    
    def _extract_features(self, query: str) -> UncertaintyFeatures:
        """Extract all uncertainty features from query."""
        
        # Tokenize query
        tokens = self._tokenize(query)
        
        # Calculate individual features
        ambiguity = self._calculate_ambiguity(tokens, query)
        specificity = self._calculate_specificity(tokens, query)  
        coverage = self._calculate_coverage(tokens)
        coherence = self._calculate_coherence(tokens, query)
        complexity = self._calculate_complexity(tokens, query)
        
        return UncertaintyFeatures(
            ambiguity_score=ambiguity,
            specificity_score=1.0 - specificity,  # Invert: low specificity = high uncertainty
            coverage_score=coverage,
            coherence_score=coherence,
            complexity_score=complexity
        )
    
    def _tokenize(self, query: str) -> List[str]:
        """Tokenize query into clean terms."""
        # Convert to lowercase and remove punctuation
        query_clean = query.lower().translate(str.maketrans('', '', string.punctuation))
        
        # Split into tokens and filter out empty/short terms
        tokens = [token.strip() for token in query_clean.split() if len(token.strip()) > 1]
        
        return tokens
    
    def _calculate_ambiguity(self, tokens: List[str], query: str) -> float:
        """
        Calculate query ambiguity score.
        
        High ambiguity indicators:
        - Contains ambiguous terms
        - Short query (lacks context)
        - Question words without specifics
        - Multiple possible interpretations
        """
        if not tokens:
            return 1.0
        
        ambiguity_factors = []
        
        # 1. Ambiguous term density
        ambiguous_count = sum(1 for token in tokens if token in self.AMBIGUOUS_TERMS)
        ambiguous_density = ambiguous_count / len(tokens)
        ambiguity_factors.append(ambiguous_density)
        
        # 2. Query length penalty (very short queries are ambiguous)
        if len(tokens) == 1:
            length_penalty = 0.9  # Single words are very ambiguous
        elif len(tokens) <= 2:
            length_penalty = 0.8
        elif len(tokens) <= 4:
            length_penalty = 0.5  # Increased penalty for short queries
        else:
            length_penalty = 0.1
        ambiguity_factors.append(length_penalty)
        
        # 3. Question words without specifics
        question_words = {'what', 'how', 'when', 'where', 'why', 'which'}
        has_question = any(token in question_words for token in tokens)
        has_specifics = any(token in self.SPECIFIC_TERMS for token in tokens)
        
        if has_question and not has_specifics:
            ambiguity_factors.append(0.7)
        
        # 4. Generic terms
        generic_terms = {'information', 'details', 'about', 'regarding', 'concerning'}
        generic_count = sum(1 for token in tokens if token in generic_terms)
        if generic_count > 0:
            ambiguity_factors.append(min(0.6, generic_count * 0.3))
        
        # 5. Single word context ambiguity (any single word lacks context)
        if len(tokens) == 1:
            ambiguity_factors.append(0.9)  # Very strong penalty for lack of context
        
        # Return average of factors (capped at 1.0)
        return min(1.0, sum(ambiguity_factors) / max(1, len(ambiguity_factors)))
    
    def _calculate_specificity(self, tokens: List[str], query: str) -> float:
        """
        Calculate query specificity score.
        
        High specificity indicators:
        - Contains specific terms
        - Longer, detailed queries
        - Numbers, codes, proper nouns
        - Precise terminology
        """
        if not tokens:
            return 0.0
        
        specificity_factors = []
        
        # 1. Specific term density
        specific_count = sum(1 for token in tokens if token in self.SPECIFIC_TERMS)
        specific_density = specific_count / len(tokens)
        specificity_factors.append(specific_density)
        
        # 2. Query length bonus (longer queries tend to be more specific)
        if len(tokens) >= 8:
            length_bonus = 0.8
        elif len(tokens) >= 5:
            length_bonus = 0.6
        elif len(tokens) >= 3:
            length_bonus = 0.4
        else:
            length_bonus = 0.1
        specificity_factors.append(length_bonus)
        
        # 3. Numbers and codes (indicate specificity)
        has_numbers = bool(re.search(r'\d', query))
        if has_numbers:
            specificity_factors.append(0.7)
        
        # 4. Capitalized terms (proper nouns, acronyms)
        capitalized_count = sum(1 for word in query.split() if word[0].isupper() and len(word) > 1)
        if capitalized_count > 0:
            specificity_factors.append(min(0.6, capitalized_count * 0.2))
        
        # 5. Technical terms (expanded list)
        technical_terms = {
            'form', 'section', 'paragraph', 'article', 'clause', 'regulation',
            'policy', 'directive', 'procedure', 'requirement', 'criteria',
            'authorization', 'approval', 'documentation', 'receipt', 'claim'
        }
        technical_count = sum(1 for token in tokens if token in technical_terms)
        if technical_count > 0:
            specificity_factors.append(min(0.7, technical_count * 0.3))  # Increased weighting
        
        # Return average of factors (capped at 1.0)
        return min(1.0, sum(specificity_factors) / max(1, len(specificity_factors)))
    
    def _calculate_coverage(self, tokens: List[str]) -> float:
        """
        Calculate domain coverage uncertainty.
        
        High coverage uncertainty = query likely outside our domain knowledge
        """
        if not tokens:
            return 1.0
        
        # Count domain-relevant terms
        domain_matches = sum(1 for token in tokens if token in self.CF_TRAVEL_TERMS)
        coverage_ratio = domain_matches / len(tokens)
        
        # Convert to uncertainty score (high coverage = low uncertainty)
        if coverage_ratio >= 0.5:
            return 0.1  # Very low uncertainty
        elif coverage_ratio >= 0.3:
            return 0.3  # Low uncertainty
        elif coverage_ratio >= 0.1:
            return 0.6  # Medium uncertainty
        else:
            return 0.9  # High uncertainty - likely out of domain
    
    def _calculate_coherence(self, tokens: List[str], query: str) -> float:
        """
        Calculate semantic coherence score.
        
        High coherence uncertainty = query parts don't fit together well
        """
        if len(tokens) == 0:
            return 0.8  # Empty token list after cleaning = incoherent
        elif len(tokens) < 2:
            return 0.2  # Single terms are coherent by definition
        
        coherence_factors = []
        
        # 1. Repetition penalty (indicates unclear thinking)
        token_counts = Counter(tokens)
        max_repetition = max(token_counts.values())
        if max_repetition > 1:
            repetition_penalty = min(0.4, (max_repetition - 1) * 0.2)
            coherence_factors.append(repetition_penalty)
        
        # 2. Mixed domain penalty
        # Check if query mixes CF travel terms with unrelated terms
        cf_terms = sum(1 for token in tokens if token in self.CF_TRAVEL_TERMS)
        if cf_terms > 0 and cf_terms < len(tokens):
            # Has some CF terms but also other terms
            other_terms = len(tokens) - cf_terms
            if other_terms > cf_terms:  # More non-CF than CF terms
                coherence_factors.append(0.3)
        
        # 3. Contradictory terms
        contradictions = [
            (['maximum', 'minimum'], 0.4),
            (['domestic', 'international'], 0.2),
            (['temporary', 'permanent'], 0.3),
        ]
        
        for contradiction_pair, penalty in contradictions:
            if all(any(term in token for token in tokens) for term in contradiction_pair):
                coherence_factors.append(penalty)
        
        # Return average uncertainty (default to low if no issues found)
        return sum(coherence_factors) / max(1, len(coherence_factors)) if coherence_factors else 0.1
    
    def _calculate_complexity(self, tokens: List[str], query: str) -> float:
        """
        Calculate query complexity score.
        
        High complexity = harder to process, more uncertainty
        """
        if not tokens:
            return 0.0
        
        complexity_factors = []
        
        # 1. Length-based complexity
        if len(tokens) > 15:
            complexity_factors.append(0.8)
        elif len(tokens) > 10:
            complexity_factors.append(0.5)
        elif len(tokens) > 7:
            complexity_factors.append(0.3)
        
        # 2. Compound questions (multiple question marks or question words)
        question_words = {'what', 'how', 'when', 'where', 'why', 'which'}
        question_count = sum(1 for token in tokens if token in question_words)
        if question_count > 1:
            complexity_factors.append(min(0.6, question_count * 0.2))
        
        # 3. Multiple clauses (AND/OR logic)
        logical_connectors = {'and', 'or', 'but', 'however', 'also', 'additionally'}
        connector_count = sum(1 for token in tokens if token in logical_connectors)
        if connector_count > 0:
            complexity_factors.append(min(0.4, connector_count * 0.2))
        
        # 4. Parentheses or quotes (structured complexity)
        if '(' in query or '"' in query or "'" in query:
            complexity_factors.append(0.3)
        
        # Return average complexity
        return sum(complexity_factors) / max(1, len(complexity_factors)) if complexity_factors else 0.1
    
    def _classify_confidence(self, uncertainty: float) -> str:
        """Classify confidence level based on overall uncertainty."""
        if uncertainty <= self.confidence_thresholds['high']:
            return 'high'
        elif uncertainty <= self.confidence_thresholds['medium']:
            return 'medium'
        else:
            return 'low'
    
    def _generate_reasoning(self, features: UncertaintyFeatures, query: str) -> List[str]:
        """Generate human-readable reasoning for the uncertainty assessment."""
        reasoning = []
        
        # Analyze each feature
        if features.ambiguity_score > 0.6:
            reasoning.append(f"High ambiguity detected (score: {features.ambiguity_score:.2f}) - query contains vague or multiple-meaning terms")
        elif features.ambiguity_score < 0.3:
            reasoning.append(f"Low ambiguity (score: {features.ambiguity_score:.2f}) - query terms are clear and unambiguous")
        
        if features.specificity_score > 0.6:  # Remember this is inverted
            reasoning.append(f"Low specificity (score: {1-features.specificity_score:.2f}) - query lacks detail and precision")
        elif features.specificity_score < 0.3:
            reasoning.append(f"High specificity (score: {1-features.specificity_score:.2f}) - query is detailed and precise")
        
        if features.coverage_score > 0.6:
            reasoning.append(f"Poor domain coverage (score: {features.coverage_score:.2f}) - query may be outside CF travel domain")
        elif features.coverage_score < 0.3:
            reasoning.append(f"Good domain coverage (score: {features.coverage_score:.2f}) - query aligns well with CF travel topics")
        
        if features.coherence_score > 0.5:
            reasoning.append(f"Coherence issues (score: {features.coherence_score:.2f}) - query parts may not fit together well")
        
        if features.complexity_score > 0.5:
            reasoning.append(f"High complexity (score: {features.complexity_score:.2f}) - query is complex and may be difficult to process")
        
        if not reasoning:
            reasoning.append("Query appears clear and well-formed with no major uncertainty factors")
        
        return reasoning
    
    def _recommend_retrievers(
        self, 
        overall_uncertainty: float, 
        features: UncertaintyFeatures
    ) -> Dict[str, bool]:
        """
        Recommend which retrievers to activate based on uncertainty analysis.
        
        Strategy:
        - High confidence: Use core retrievers only (dense + sparse)
        - Medium confidence: Add BM25 for additional coverage
        - Low confidence: Activate all retrievers for maximum recall
        """
        recommendations = {
            'dense': True,   # Always use dense retrieval
            'sparse': True,  # Always use sparse retrieval  
            'bm25': False,   # Optional keyword matching
            'hybrid': False  # Optional advanced hybrid methods
        }
        
        # Activate additional retrievers based on uncertainty
        if overall_uncertainty > 0.6:  # Low confidence
            recommendations['bm25'] = True
            recommendations['hybrid'] = True
        elif overall_uncertainty > 0.3:  # Medium confidence
            recommendations['bm25'] = True
        
        # Special cases based on feature analysis
        if features.coverage_score > 0.7:  # Out-of-domain
            recommendations['bm25'] = True  # BM25 good for edge cases
            
        if features.ambiguity_score > 0.7:  # Very ambiguous
            recommendations['hybrid'] = True  # Need all methods
            
        if features.complexity_score > 0.6:  # Complex query
            recommendations['bm25'] = True  # Keyword matching helps
        
        return recommendations
    
    def _create_empty_result(self) -> UncertaintyResult:
        """Create result for empty/invalid queries."""
        empty_features = UncertaintyFeatures(
            ambiguity_score=1.0,
            specificity_score=1.0, 
            coverage_score=1.0,
            coherence_score=1.0,
            complexity_score=0.0
        )
        
        return UncertaintyResult(
            overall_uncertainty=1.0,
            confidence_level='low',
            features=empty_features,
            reasoning=["Empty or invalid query"],
            retriever_recommendations={
                'dense': False,
                'sparse': False,
                'bm25': False,
                'hybrid': False
            }
        )


def create_uncertainty_scorer(
    weights: Optional[Dict[str, float]] = None,
    confidence_thresholds: Optional[Dict[str, float]] = None
) -> UncertaintyScorer:
    """
    Factory function to create uncertainty scorer with default settings.
    
    Args:
        weights: Custom feature weights
        confidence_thresholds: Custom confidence thresholds
        
    Returns:
        Configured UncertaintyScorer instance
    """
    return UncertaintyScorer(
        weights=weights,
        confidence_thresholds=confidence_thresholds
    )