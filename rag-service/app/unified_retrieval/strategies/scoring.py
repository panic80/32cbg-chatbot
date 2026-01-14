"""Scoring strategies for the unified retrieval framework."""

from typing import List, Dict, Any, Optional, Tuple
import re
from datetime import datetime
import numpy as np

from langchain_core.documents import Document

from app.unified_retrieval.strategies.base import BaseStrategy, RetrievalContext, StrategyType
from app.core.logging import get_logger

logger = get_logger(__name__)


class ContentBoostStrategy(BaseStrategy):
    """
    Boost based on content patterns.
    
    This strategy boosts documents based on specific content patterns
    like tables, lists, specific values, or authoritative language.
    """
    
    def __init__(
        self,
        pattern_boosts: Optional[Dict[str, float]] = None,
        **kwargs
    ):
        """
        Initialize the content boost strategy.
        
        Args:
            pattern_boosts: Dictionary of patterns and their boost values
        """
        super().__init__(
            strategy_type=StrategyType.SCORING,
            name="content_boost_strategy",
            required_inputs=["documents"],
            **kwargs
        )
        self.pattern_boosts = pattern_boosts or self._get_default_boosts()
    
    def _get_default_boosts(self) -> Dict[str, float]:
        """Get default content pattern boosts."""
        return {
            "table": 1.5,  # Contains tables
            "list": 1.2,   # Contains lists
            "rate": 1.3,   # Contains specific rates/values
            "procedure": 1.2,  # Contains procedures
            "definition": 1.1,  # Contains definitions
            "example": 1.1,  # Contains examples
            "official": 1.4,  # Official/authoritative language
            "current": 1.3,  # Current/up-to-date information
        }
    
    def _detect_patterns(self, content: str) -> Dict[str, bool]:
        """Detect content patterns in document."""
        content_lower = content.lower()
        
        patterns = {
            "table": bool(re.search(r'\|.*\|.*\||\t.*\t|table\s*\d+|column|row', content_lower)),
            "list": bool(re.search(r'^\s*[\d\-\*]\s+|\n\s*[a-z]\)', content, re.MULTILINE)),
            "rate": bool(re.search(r'\$\s*\d+|\d+\s*per\s*(day|km|meal)|\d+%', content_lower)),
            "procedure": bool(re.search(r'step\s*\d+|procedure|process|follow these', content_lower)),
            "definition": bool(re.search(r'means|refers to|is defined as|definition', content_lower)),
            "example": bool(re.search(r'for example|e\.g\.|such as|instance', content_lower)),
            "official": bool(re.search(r'shall|must|required|authorized|directive|policy', content_lower)),
            "current": bool(re.search(r'effective|current|as of|updated|latest', content_lower)),
        }
        
        return patterns
    
    def _calculate_content_score(self, doc: Document) -> float:
        """Calculate content-based score for document."""
        patterns = self._detect_patterns(doc.page_content)
        
        # Calculate boost multiplier
        boost = 1.0
        detected_patterns = []
        
        for pattern_name, is_present in patterns.items():
            if is_present and pattern_name in self.pattern_boosts:
                boost *= self.pattern_boosts[pattern_name]
                detected_patterns.append(pattern_name)
        
        # Store detected patterns in metadata
        doc.metadata["detected_patterns"] = detected_patterns
        
        return boost
    
    async def execute(self, context: RetrievalContext) -> RetrievalContext:
        """Execute the content boost scoring."""
        documents = context.documents
        
        # Calculate content scores
        for doc in documents:
            content_boost = self._calculate_content_score(doc)
            
            # Apply boost to existing score or set initial score
            current_score = context.scores.get(id(doc), 1.0)
            new_score = current_score * content_boost
            context.scores[id(doc)] = new_score
            
            # Store boost in metadata
            if "score_boosts" not in doc.metadata:
                doc.metadata["score_boosts"] = {}
            doc.metadata["score_boosts"]["content"] = content_boost
        
        # Add strategy output
        context.add_strategy_output(
            self.component_name,
            {
                "docs_scored": len(documents),
                "average_boost": np.mean([
                    doc.metadata.get("score_boosts", {}).get("content", 1.0)
                    for doc in documents
                ]) if documents else 1.0
            }
        )
        
        return context


class AuthorityBoostStrategy(BaseStrategy):
    """
    Boost official/authoritative sources.
    
    This strategy boosts documents from official sources or
    those with high authority indicators.
    """
    
    def __init__(
        self,
        authority_sources: Optional[Dict[str, float]] = None,
        authority_indicators: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize the authority boost strategy.
        
        Args:
            authority_sources: Dictionary of source names and their authority scores
            authority_indicators: List of terms indicating authority
        """
        super().__init__(
            strategy_type=StrategyType.SCORING,
            name="authority_boost_strategy",
            required_inputs=["documents"],
            **kwargs
        )
        self.authority_sources = authority_sources or self._get_default_sources()
        self.authority_indicators = authority_indicators or self._get_default_indicators()
    
    def _get_default_sources(self) -> Dict[str, float]:
        """Get default authoritative sources."""
        return {
            "cbi": 2.0,  # Compensation and Benefits Instructions
            "qr&o": 1.8,  # Queen's Regulations and Orders
            "cfao": 1.7,  # Canadian Forces Administrative Orders
            "daod": 1.7,  # Defence Administrative Orders and Directives
            "canforgen": 1.6,  # Canadian Forces General Messages
            "cftdi": 2.0,  # Canadian Forces TDI
            "official": 1.5,  # Generic official documents
        }
    
    def _get_default_indicators(self) -> List[str]:
        """Get default authority indicators."""
        return [
            "director general", "chief of", "minister",
            "headquarters", "official", "directive",
            "shall", "must", "required", "mandatory",
            "policy", "regulation", "instruction"
        ]
    
    def _calculate_authority_score(self, doc: Document) -> float:
        """Calculate authority score for document."""
        boost = 1.0
        
        # Check source authority
        source = doc.metadata.get("source", "").lower()
        for auth_source, score in self.authority_sources.items():
            if auth_source in source:
                boost = max(boost, score)
                break
        
        # Check content for authority indicators
        content_lower = doc.page_content.lower()
        indicator_count = sum(
            1 for indicator in self.authority_indicators
            if indicator.lower() in content_lower
        )
        
        # Add indicator boost (diminishing returns)
        if indicator_count > 0:
            indicator_boost = 1 + (0.1 * min(indicator_count, 5))
            boost *= indicator_boost
        
        # Check metadata for official flags
        if doc.metadata.get("is_official", False):
            boost *= 1.5
        
        # Check for recent effective dates
        effective_date = doc.metadata.get("effective_date")
        if effective_date:
            try:
                # Parse date and check recency
                date = datetime.fromisoformat(effective_date.replace("Z", "+00:00"))
                years_old = (datetime.now() - date.replace(tzinfo=None)).days / 365
                if years_old < 1:
                    boost *= 1.2  # Very recent
                elif years_old < 3:
                    boost *= 1.1  # Recent
            except:
                pass
        
        return boost
    
    async def execute(self, context: RetrievalContext) -> RetrievalContext:
        """Execute the authority boost scoring."""
        documents = context.documents
        
        # Calculate authority scores
        for doc in documents:
            authority_boost = self._calculate_authority_score(doc)
            
            # Apply boost to existing score
            current_score = context.scores.get(id(doc), 1.0)
            new_score = current_score * authority_boost
            context.scores[id(doc)] = new_score
            
            # Store boost in metadata
            if "score_boosts" not in doc.metadata:
                doc.metadata["score_boosts"] = {}
            doc.metadata["score_boosts"]["authority"] = authority_boost
        
        # Add strategy output
        context.add_strategy_output(
            self.component_name,
            {
                "docs_scored": len(documents),
                "authority_boosts_applied": sum(
                    1 for doc in documents
                    if doc.metadata.get("score_boosts", {}).get("authority", 1.0) > 1.0
                )
            }
        )
        
        return context


class CooccurrenceScoreStrategy(BaseStrategy):
    """
    Score based on concept relationships.
    
    This strategy scores documents based on how well they contain
    co-occurring concepts from the query.
    """
    
    def __init__(
        self,
        min_concept_length: int = 3,
        cooccurrence_window: int = 50,
        **kwargs
    ):
        """
        Initialize the co-occurrence score strategy.
        
        Args:
            min_concept_length: Minimum word length to consider as concept
            cooccurrence_window: Word window for co-occurrence
        """
        super().__init__(
            strategy_type=StrategyType.SCORING,
            name="cooccurrence_score_strategy",
            required_inputs=["documents"],
            **kwargs
        )
        self.min_concept_length = min_concept_length
        self.cooccurrence_window = cooccurrence_window
    
    def _extract_concepts(self, text: str) -> List[str]:
        """Extract concept words from text."""
        # Simple concept extraction: significant words
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter stop words and short words
        stop_words = {'the', 'is', 'at', 'which', 'on', 'and', 'a', 'an', 'as', 'are', 'was', 'were', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'could', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'them', 'their', 'what', 'which', 'who', 'when', 'where', 'why', 'how', 'all', 'each', 'every', 'some', 'any', 'few', 'many', 'much', 'most', 'other', 'another', 'such', 'no', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'but', 'or', 'if', 'then', 'else', 'because', 'while', 'of', 'for', 'with', 'about', 'between', 'through', 'during', 'before', 'after', 'below', 'above', 'up', 'down', 'in', 'out', 'off', 'over', 'under', 'from', 'to', 'into', 'by'}
        
        concepts = [
            w for w in words
            if len(w) >= self.min_concept_length and w not in stop_words
        ]
        
        return concepts
    
    def _calculate_cooccurrence_score(self, doc: Document, query_concepts: List[str]) -> float:
        """Calculate co-occurrence score for document."""
        if not query_concepts:
            return 1.0
        
        doc_words = doc.page_content.lower().split()
        
        # Find positions of query concepts in document
        concept_positions = {}
        for i, word in enumerate(doc_words):
            for concept in query_concepts:
                if concept in word:
                    if concept not in concept_positions:
                        concept_positions[concept] = []
                    concept_positions[concept].append(i)
        
        # Calculate co-occurrence score
        if len(concept_positions) < 2:
            # Not enough concepts found
            return 1.0 + (0.1 * len(concept_positions))
        
        # Check how many concepts appear close together
        cooccurrence_count = 0
        concepts_found = list(concept_positions.keys())
        
        for i in range(len(concepts_found)):
            for j in range(i + 1, len(concepts_found)):
                concept1 = concepts_found[i]
                concept2 = concepts_found[j]
                
                # Check if any positions are within window
                for pos1 in concept_positions[concept1]:
                    for pos2 in concept_positions[concept2]:
                        if abs(pos1 - pos2) <= self.cooccurrence_window:
                            cooccurrence_count += 1
                            break
        
        # Calculate score based on co-occurrences
        base_score = len(concept_positions) / len(query_concepts)
        cooccurrence_bonus = min(cooccurrence_count * 0.2, 1.0)
        
        return 1.0 + base_score + cooccurrence_bonus
    
    async def execute(self, context: RetrievalContext) -> RetrievalContext:
        """Execute the co-occurrence scoring."""
        documents = context.documents
        query = context.get_query()
        
        # Extract query concepts
        query_concepts = self._extract_concepts(query)
        
        # Calculate co-occurrence scores
        for doc in documents:
            cooccurrence_score = self._calculate_cooccurrence_score(doc, query_concepts)
            
            # Apply score
            current_score = context.scores.get(id(doc), 1.0)
            new_score = current_score * cooccurrence_score
            context.scores[id(doc)] = new_score
            
            # Store in metadata
            if "score_boosts" not in doc.metadata:
                doc.metadata["score_boosts"] = {}
            doc.metadata["score_boosts"]["cooccurrence"] = cooccurrence_score
        
        # Add strategy output
        context.add_strategy_output(
            self.component_name,
            {
                "query_concepts": query_concepts,
                "docs_scored": len(documents),
                "concepts_used": len(query_concepts)
            }
        )
        
        return context


class HybridScoreStrategy(BaseStrategy):
    """
    Combine multiple scoring methods.
    
    This strategy combines different scoring approaches (e.g., vector similarity,
    BM25, custom scores) into a unified score.
    """
    
    def __init__(
        self,
        score_weights: Optional[Dict[str, float]] = None,
        normalization: str = "minmax",
        **kwargs
    ):
        """
        Initialize the hybrid score strategy.
        
        Args:
            score_weights: Weights for different score types
            normalization: Normalization method ('minmax', 'zscore', 'none')
        """
        super().__init__(
            strategy_type=StrategyType.SCORING,
            name="hybrid_score_strategy",
            required_inputs=["documents"],
            **kwargs
        )
        self.score_weights = score_weights or self._get_default_weights()
        self.normalization = normalization
    
    def _get_default_weights(self) -> Dict[str, float]:
        """Get default score weights."""
        return {
            "vector_similarity": 0.4,
            "bm25": 0.3,
            "content_boost": 0.15,
            "authority_boost": 0.1,
            "cooccurrence": 0.05
        }
    
    def _normalize_scores(self, scores: List[float], method: str) -> List[float]:
        """Normalize a list of scores."""
        if not scores or method == "none":
            return scores
        
        scores_array = np.array(scores)
        
        if method == "minmax":
            min_score = scores_array.min()
            max_score = scores_array.max()
            if max_score > min_score:
                return ((scores_array - min_score) / (max_score - min_score)).tolist()
            else:
                return [0.5] * len(scores)
        
        elif method == "zscore":
            mean = scores_array.mean()
            std = scores_array.std()
            if std > 0:
                return ((scores_array - mean) / std).tolist()
            else:
                return [0.0] * len(scores)
        
        return scores
    
    def _extract_score_components(self, doc: Document) -> Dict[str, float]:
        """Extract different score components from document metadata."""
        components = {}
        
        # Vector similarity score
        if "similarity_score" in doc.metadata:
            components["vector_similarity"] = doc.metadata["similarity_score"]
        
        # BM25 score
        if "bm25_score" in doc.metadata:
            components["bm25"] = doc.metadata["bm25_score"]
        
        # Extract boost scores
        boosts = doc.metadata.get("score_boosts", {})
        for boost_type, boost_value in boosts.items():
            components[f"{boost_type}_boost"] = boost_value
        
        return components
    
    async def execute(self, context: RetrievalContext) -> RetrievalContext:
        """Execute the hybrid scoring."""
        documents = context.documents
        
        if not documents:
            return context
        
        # Collect all score components
        all_components = {}
        for doc in documents:
            doc_components = self._extract_score_components(doc)
            for component_name, score in doc_components.items():
                if component_name not in all_components:
                    all_components[component_name] = []
                all_components[component_name].append(score)
        
        # Normalize each component
        normalized_components = {}
        for component_name, scores in all_components.items():
            normalized_components[component_name] = self._normalize_scores(
                scores, self.normalization
            )
        
        # Calculate hybrid scores
        for i, doc in enumerate(documents):
            hybrid_score = 0.0
            score_breakdown = {}
            
            for component_name, weight in self.score_weights.items():
                if component_name in normalized_components:
                    component_score = normalized_components[component_name][i]
                    weighted_score = component_score * weight
                    hybrid_score += weighted_score
                    score_breakdown[component_name] = {
                        "raw": all_components.get(component_name, [0])[i],
                        "normalized": component_score,
                        "weighted": weighted_score
                    }
            
            # Store hybrid score
            context.scores[id(doc)] = hybrid_score
            doc.metadata["hybrid_score"] = hybrid_score
            doc.metadata["score_breakdown"] = score_breakdown
        
        # Sort documents by hybrid score
        sorted_docs = sorted(
            documents,
            key=lambda d: context.scores.get(id(d), 0),
            reverse=True
        )
        context.documents = sorted_docs
        
        # Add strategy output
        context.add_strategy_output(
            self.component_name,
            {
                "docs_scored": len(documents),
                "score_components": list(all_components.keys()),
                "normalization": self.normalization,
                "weights": self.score_weights
            }
        )
        
        return context