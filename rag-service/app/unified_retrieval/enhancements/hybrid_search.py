"""Enhanced Hybrid Search with Advanced Fusion Techniques

This module implements sophisticated hybrid search combining dense and sparse
retrieval with advanced fusion strategies including Reciprocal Rank Fusion (RRF).
"""

import re
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from langchain_core.documents import Document
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class EnhancedHybridSearch:
    """Advanced hybrid search with multiple fusion strategies"""
    
    def __init__(
        self,
        rrf_k: int = 60,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        normalization_method: str = "minmax",
        fusion_strategy: str = "rrf"
    ):
        """Initialize enhanced hybrid search
        
        Args:
            rrf_k: Parameter for RRF fusion (typically 60)
            dense_weight: Weight for dense retrieval scores
            sparse_weight: Weight for sparse retrieval scores
            normalization_method: Score normalization method
            fusion_strategy: Fusion strategy to use
        """
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.normalization_method = normalization_method
        self.fusion_strategy = fusion_strategy
        
        # Query type weights
        self.query_type_weights = {
            "keyword": {"dense": 0.3, "sparse": 0.7},
            "semantic": {"dense": 0.8, "sparse": 0.2},
            "hybrid": {"dense": 0.5, "sparse": 0.5},
            "exact": {"dense": 0.2, "sparse": 0.8},
            "conceptual": {"dense": 0.9, "sparse": 0.1}
        }
    
    def fuse_results(
        self,
        dense_results: List[Tuple[Document, float]],
        sparse_results: List[Tuple[Document, float]],
        query_type: Optional[str] = None,
        strategy: Optional[str] = None
    ) -> List[Tuple[Document, float]]:
        """Fuse dense and sparse retrieval results
        
        Args:
            dense_results: List of (document, score) from dense retrieval
            sparse_results: List of (document, score) from sparse retrieval
            query_type: Type of query for weight adjustment
            strategy: Fusion strategy override
            
        Returns:
            Fused and reranked results
        """
        strategy = strategy or self.fusion_strategy
        
        # Adjust weights based on query type
        if query_type and query_type in self.query_type_weights:
            weights = self.query_type_weights[query_type]
            dense_weight = weights["dense"]
            sparse_weight = weights["sparse"]
        else:
            dense_weight = self.dense_weight
            sparse_weight = self.sparse_weight
        
        # Apply fusion strategy
        if strategy == "rrf":
            return self._reciprocal_rank_fusion(
                dense_results, 
                sparse_results,
                dense_weight,
                sparse_weight
            )
        elif strategy == "weighted":
            return self._weighted_fusion(
                dense_results,
                sparse_results,
                dense_weight,
                sparse_weight
            )
        elif strategy == "combsum":
            return self._combsum_fusion(
                dense_results,
                sparse_results,
                dense_weight,
                sparse_weight
            )
        elif strategy == "combmnz":
            return self._combmnz_fusion(
                dense_results,
                sparse_results,
                dense_weight,
                sparse_weight
            )
        else:
            raise ValueError(f"Unknown fusion strategy: {strategy}")
    
    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Tuple[Document, float]],
        sparse_results: List[Tuple[Document, float]],
        dense_weight: float,
        sparse_weight: float
    ) -> List[Tuple[Document, float]]:
        """Reciprocal Rank Fusion (RRF) for combining results"""
        # Create document to score mapping
        rrf_scores = defaultdict(float)
        doc_map = {}
        
        # Process dense results
        for rank, (doc, score) in enumerate(dense_results):
            doc_id = self._get_doc_id(doc)
            doc_map[doc_id] = doc
            rrf_scores[doc_id] += dense_weight / (self.rrf_k + rank + 1)
        
        # Process sparse results
        for rank, (doc, score) in enumerate(sparse_results):
            doc_id = self._get_doc_id(doc)
            doc_map[doc_id] = doc
            rrf_scores[doc_id] += sparse_weight / (self.rrf_k + rank + 1)
        
        # Sort by RRF score
        sorted_results = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Return documents with scores
        return [
            (doc_map[doc_id], score)
            for doc_id, score in sorted_results
        ]
    
    def _weighted_fusion(
        self,
        dense_results: List[Tuple[Document, float]],
        sparse_results: List[Tuple[Document, float]],
        dense_weight: float,
        sparse_weight: float
    ) -> List[Tuple[Document, float]]:
        """Weighted score fusion with normalization"""
        # Normalize scores
        dense_normalized = self._normalize_scores(dense_results)
        sparse_normalized = self._normalize_scores(sparse_results)
        
        # Create score mapping
        weighted_scores = defaultdict(float)
        doc_map = {}
        
        # Add dense scores
        for doc, score in dense_normalized:
            doc_id = self._get_doc_id(doc)
            doc_map[doc_id] = doc
            weighted_scores[doc_id] += dense_weight * score
        
        # Add sparse scores
        for doc, score in sparse_normalized:
            doc_id = self._get_doc_id(doc)
            doc_map[doc_id] = doc
            weighted_scores[doc_id] += sparse_weight * score
        
        # Sort by weighted score
        sorted_results = sorted(
            weighted_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [
            (doc_map[doc_id], score)
            for doc_id, score in sorted_results
        ]
    
    def _combsum_fusion(
        self,
        dense_results: List[Tuple[Document, float]],
        sparse_results: List[Tuple[Document, float]],
        dense_weight: float,
        sparse_weight: float
    ) -> List[Tuple[Document, float]]:
        """CombSUM fusion - sum of normalized scores"""
        # Normalize scores
        dense_normalized = self._normalize_scores(dense_results)
        sparse_normalized = self._normalize_scores(sparse_results)
        
        # Create score mapping
        sum_scores = defaultdict(float)
        doc_map = {}
        
        # Sum scores
        for doc, score in dense_normalized:
            doc_id = self._get_doc_id(doc)
            doc_map[doc_id] = doc
            sum_scores[doc_id] += score * dense_weight
        
        for doc, score in sparse_normalized:
            doc_id = self._get_doc_id(doc)
            doc_map[doc_id] = doc
            sum_scores[doc_id] += score * sparse_weight
        
        # Sort by sum
        sorted_results = sorted(
            sum_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [
            (doc_map[doc_id], score)
            for doc_id, score in sorted_results
        ]
    
    def _combmnz_fusion(
        self,
        dense_results: List[Tuple[Document, float]],
        sparse_results: List[Tuple[Document, float]],
        dense_weight: float,
        sparse_weight: float
    ) -> List[Tuple[Document, float]]:
        """CombMNZ fusion - sum multiplied by number of non-zero scores"""
        # Normalize scores
        dense_normalized = self._normalize_scores(dense_results)
        sparse_normalized = self._normalize_scores(sparse_results)
        
        # Track scores and non-zero counts
        sum_scores = defaultdict(float)
        nonzero_counts = defaultdict(int)
        doc_map = {}
        
        # Process dense results
        for doc, score in dense_normalized:
            doc_id = self._get_doc_id(doc)
            doc_map[doc_id] = doc
            if score > 0:
                sum_scores[doc_id] += score * dense_weight
                nonzero_counts[doc_id] += 1
        
        # Process sparse results
        for doc, score in sparse_normalized:
            doc_id = self._get_doc_id(doc)
            doc_map[doc_id] = doc
            if score > 0:
                sum_scores[doc_id] += score * sparse_weight
                nonzero_counts[doc_id] += 1
        
        # Calculate CombMNZ scores
        mnz_scores = {
            doc_id: sum_scores[doc_id] * nonzero_counts[doc_id]
            for doc_id in sum_scores
        }
        
        # Sort by MNZ score
        sorted_results = sorted(
            mnz_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [
            (doc_map[doc_id], score)
            for doc_id, score in sorted_results
        ]
    
    def _normalize_scores(
        self,
        results: List[Tuple[Document, float]]
    ) -> List[Tuple[Document, float]]:
        """Normalize scores using specified method"""
        if not results:
            return results
        
        scores = [score for _, score in results]
        
        if self.normalization_method == "minmax":
            min_score = min(scores)
            max_score = max(scores)
            if max_score == min_score:
                normalized = [1.0] * len(scores)
            else:
                normalized = [
                    (score - min_score) / (max_score - min_score)
                    for score in scores
                ]
        
        elif self.normalization_method == "zscore":
            mean = np.mean(scores)
            std = np.std(scores)
            if std == 0:
                normalized = [0.5] * len(scores)
            else:
                normalized = [
                    (score - mean) / std
                    for score in scores
                ]
                # Convert to 0-1 range
                min_norm = min(normalized)
                max_norm = max(normalized)
                if max_norm > min_norm:
                    normalized = [
                        (n - min_norm) / (max_norm - min_norm)
                        for n in normalized
                    ]
        
        elif self.normalization_method == "rank":
            # Rank-based normalization
            rank_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
            normalized = [0.0] * len(scores)
            for rank, (idx, _) in enumerate(rank_scores):
                normalized[idx] = 1.0 - (rank / len(scores))
        
        else:
            # No normalization
            normalized = scores
        
        return [
            (doc, norm_score)
            for (doc, _), norm_score in zip(results, normalized)
        ]
    
    def _get_doc_id(self, doc: Document) -> str:
        """Generate unique document ID"""
        # Use metadata ID if available
        if "id" in doc.metadata:
            return doc.metadata["id"]
        
        # Otherwise create hash from content
        import hashlib
        content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
        return content_hash
    
    def analyze_query_type(self, query: str) -> str:
        """Analyze query to determine optimal fusion weights
        
        Returns:
            Query type: 'keyword', 'semantic', 'hybrid', 'exact', 'conceptual'
        """
        query_lower = query.lower()
        
        # Exact match patterns (quotes, specific identifiers)
        if '"' in query or re.search(r'\b(cf|dnd|form)\s*\d+\b', query_lower):
            return "exact"
        
        # Keyword patterns (short, specific terms)
        if len(query.split()) <= 3 and not any(
            word in query_lower for word in ["what", "how", "why", "when", "explain"]
        ):
            return "keyword"
        
        # Conceptual patterns (abstract questions)
        if any(word in query_lower for word in ["concept", "principle", "theory", "explain", "understand"]):
            return "conceptual"
        
        # Semantic patterns (natural questions)
        if any(word in query_lower for word in ["what", "how", "why", "when", "where"]):
            return "semantic"
        
        # Default to hybrid
        return "hybrid"
    
    def get_fusion_config(self, query_type: str) -> Dict[str, Any]:
        """Get optimal fusion configuration for query type"""
        configs = {
            "keyword": {
                "strategy": "rrf",
                "dense_weight": 0.3,
                "sparse_weight": 0.7,
                "rrf_k": 40
            },
            "semantic": {
                "strategy": "weighted",
                "dense_weight": 0.8,
                "sparse_weight": 0.2,
                "normalization": "minmax"
            },
            "exact": {
                "strategy": "combmnz",
                "dense_weight": 0.2,
                "sparse_weight": 0.8
            },
            "conceptual": {
                "strategy": "rrf",
                "dense_weight": 0.9,
                "sparse_weight": 0.1,
                "rrf_k": 100
            },
            "hybrid": {
                "strategy": "rrf",
                "dense_weight": 0.5,
                "sparse_weight": 0.5,
                "rrf_k": 60
            }
        }
        
        return configs.get(query_type, configs["hybrid"])
