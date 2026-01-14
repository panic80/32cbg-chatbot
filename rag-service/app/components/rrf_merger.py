"""
Reciprocal Rank Fusion (RRF) merger for combining multiple retriever results.

This component implements the RRF algorithm to merge documents from multiple retrievers
while preserving recall and improving relevance scoring.

RRF Formula: score_RRF(d) = Σ_i 1/(k + rank_i(d))
Where k is typically between 60-120 for optimal recall preservation.
"""

import time
from typing import List, Dict, Any, Tuple, Optional, Set
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from langchain_core.documents import Document

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RRFDocument:
    """Document with RRF scoring metadata."""
    document: Document
    rrf_score: float
    retriever_ranks: Dict[str, int] = field(default_factory=dict)
    retriever_scores: Dict[str, float] = field(default_factory=dict)
    
    @property
    def id(self) -> str:
        """Get document ID from metadata or generate one."""
        return self.document.metadata.get("id", hash(self.document.page_content))


@dataclass 
class RRFMergerStats:
    """Statistics from RRF merging operation."""
    total_docs_input: int
    unique_docs_output: int
    retrievers_count: int
    merge_time_ms: float
    k_parameter: int
    score_threshold: float
    retriever_contributions: Dict[str, int] = field(default_factory=dict)
    filtered_below_threshold: int = 0


class RRFMerger:
    """
    Reciprocal Rank Fusion merger for combining results from multiple retrievers.
    
    The RRF algorithm is particularly effective for combining heterogeneous retrieval
    methods (e.g., dense + sparse retrievers) while being parameter-light and robust.
    """
    
    def __init__(
        self,
        k: int = 60,
        normalize_scores: bool = True,
        score_threshold: float = 0.0
    ):
        """
        Initialize RRF merger.
        
        Args:
            k: RRF parameter (60-120 recommended for recall preservation)
            normalize_scores: Whether to normalize final RRF scores to [0,1]
            score_threshold: Minimum normalized RRF score to keep documents (0-1)
        """
        self.k = k
        self.normalize_scores = normalize_scores
        self.score_threshold = score_threshold
        self._stats: Optional[RRFMergerStats] = None
        
        # Validate k parameter
        if not (10 <= k <= 200):
            logger.warning(f"RRF k parameter {k} outside recommended range [10, 200]")
        
        # Validate threshold
        if not (0.0 <= score_threshold <= 1.0):
            raise ValueError("score_threshold must be between 0.0 and 1.0")
        
    def merge(
        self, 
        retriever_results: Dict[str, List[Document]], 
        max_docs: Optional[int] = None
    ) -> Tuple[List[RRFDocument], RRFMergerStats]:
        """
        Merge documents from multiple retrievers using RRF.
        
        Args:
            retriever_results: Dict mapping retriever name to its document results
            max_docs: Maximum number of documents to return (None = no limit)
            
        Returns:
            Tuple of (merged_documents, merge_statistics)
        """
        start_time = time.time()
        
        # Input validation
        if not retriever_results:
            return [], self._create_empty_stats()
        
        # Remove empty retriever results
        retriever_results = {
            name: docs for name, docs in retriever_results.items() 
            if docs and len(docs) > 0
        }
        
        if not retriever_results:
            return [], self._create_empty_stats()
        
        logger.info(f"Merging results from {len(retriever_results)} retrievers: {list(retriever_results.keys())}")
        
        # Calculate RRF scores for each unique document
        doc_scores = defaultdict(lambda: {
            'score': 0.0, 
            'ranks': {}, 
            'retriever_scores': {},
            'document': None
        })
        
        total_input_docs = 0
        retriever_contributions = defaultdict(int)
        
        # Process each retriever's results
        for retriever_name, documents in retriever_results.items():
            total_input_docs += len(documents)
            retriever_contributions[retriever_name] = len(documents)
            
            for rank, doc in enumerate(documents):
                doc_id = self._get_document_id(doc)
                
                # Calculate RRF contribution from this retriever
                rrf_contribution = 1.0 / (self.k + rank + 1)  # +1 because rank is 0-indexed
                
                # Accumulate RRF score
                doc_scores[doc_id]['score'] += rrf_contribution
                doc_scores[doc_id]['ranks'][retriever_name] = rank
                
                # Store original retriever score if available
                original_score = doc.metadata.get('score', 0.0)
                doc_scores[doc_id]['retriever_scores'][retriever_name] = original_score
                
                # Store document (use first occurrence)
                if doc_scores[doc_id]['document'] is None:
                    doc_scores[doc_id]['document'] = doc
        
        # Convert to RRFDocument objects
        rrf_documents = []
        for doc_id, score_info in doc_scores.items():
            rrf_doc = RRFDocument(
                document=score_info['document'],
                rrf_score=score_info['score'],
                retriever_ranks=score_info['ranks'],
                retriever_scores=score_info['retriever_scores']
            )
            rrf_documents.append(rrf_doc)
        
        # Sort by RRF score (descending)
        rrf_documents.sort(key=lambda x: x.rrf_score, reverse=True)
        
        # Apply max_docs limit if specified
        if max_docs is not None:
            rrf_documents = rrf_documents[:max_docs]
        
        # Normalize scores if requested
        if self.normalize_scores and rrf_documents:
            max_score = rrf_documents[0].rrf_score
            min_score = rrf_documents[-1].rrf_score
            
            # Avoid division by zero - if all scores are equal, set them all to 1.0
            if max_score > min_score:
                for doc in rrf_documents:
                    doc.rrf_score = (doc.rrf_score - min_score) / (max_score - min_score)
            else:
                # All scores are equal, set the highest to 1.0 and others relative
                for i, doc in enumerate(rrf_documents):
                    doc.rrf_score = 1.0 if i == 0 else 1.0 - (i * 0.01)  # Small decrements
        
        filtered_below_threshold = 0
        
        # Apply score threshold filtering if configured
        if self.score_threshold > 0.0 and rrf_documents:
            filtered_docs = [doc for doc in rrf_documents if doc.rrf_score >= self.score_threshold]
            filtered_below_threshold = len(rrf_documents) - len(filtered_docs)
            
            # Always keep at least the top document to avoid empty results
            if not filtered_docs and rrf_documents:
                filtered_docs = [rrf_documents[0]]
                filtered_below_threshold = len(rrf_documents) - 1
            
            rrf_documents = filtered_docs
            
            if filtered_below_threshold > 0:
                logger.debug(
                    "Filtered %d documents below RRF score threshold %.2f",
                    filtered_below_threshold,
                    self.score_threshold
                )
        
        # Attach RRF metadata to documents for downstream use
        for rank, rrf_doc in enumerate(rrf_documents):
            metadata = dict(rrf_doc.document.metadata or {})
            metadata["rrf_score"] = rrf_doc.rrf_score
            metadata["rrf_rank"] = rank
            metadata["rrf_retriever_ranks"] = rrf_doc.retriever_ranks
            metadata["rrf_retriever_scores"] = rrf_doc.retriever_scores
            metadata["rrf_retrievers"] = list(rrf_doc.retriever_ranks.keys())
            rrf_doc.document.metadata = metadata
        
        # Calculate merge time
        merge_time_ms = (time.time() - start_time) * 1000
        
        # Create statistics
        stats = RRFMergerStats(
            total_docs_input=total_input_docs,
            unique_docs_output=len(rrf_documents),
            retrievers_count=len(retriever_results),
            merge_time_ms=merge_time_ms,
            k_parameter=self.k,
            score_threshold=self.score_threshold,
            retriever_contributions=dict(retriever_contributions),
            filtered_below_threshold=filtered_below_threshold
        )
        
        self._stats = stats
        
        logger.info(f"RRF merge completed: {total_input_docs} input docs → "
                   f"{len(rrf_documents)} unique docs in {merge_time_ms:.2f}ms")
        
        return rrf_documents, stats
    
    def merge_simple(
        self, 
        retriever_results: Dict[str, List[Document]], 
        max_docs: Optional[int] = None
    ) -> List[Document]:
        """
        Simplified merge that returns just the documents (for compatibility).
        
        Args:
            retriever_results: Dict mapping retriever name to its document results
            max_docs: Maximum number of documents to return
            
        Returns:
            List of merged documents sorted by RRF score
        """
        rrf_docs, _ = self.merge(retriever_results, max_docs)
        return [rrf_doc.document for rrf_doc in rrf_docs]
    
    def get_last_merge_stats(self) -> Optional[RRFMergerStats]:
        """Get statistics from the last merge operation."""
        return self._stats
    
    def _get_document_id(self, doc: Document) -> str:
        """Extract or generate a unique ID for a document."""
        # Try various metadata fields for ID
        for id_field in ['id', 'doc_id', 'document_id', 'source_id']:
            if id_field in doc.metadata and doc.metadata[id_field]:
                return str(doc.metadata[id_field])
        
        # Fallback to content hash
        content_hash = hash(doc.page_content)
        
        # Include some metadata in hash for better uniqueness
        metadata_str = str(sorted(doc.metadata.items())) if doc.metadata else ""
        combined_hash = hash(f"{content_hash}_{metadata_str}")
        
        return f"doc_{abs(combined_hash)}"
    
    def _create_empty_stats(self) -> RRFMergerStats:
        """Create empty statistics object."""
        return RRFMergerStats(
            total_docs_input=0,
            unique_docs_output=0,
            retrievers_count=0,
            merge_time_ms=0.0,
            k_parameter=self.k,
            score_threshold=self.score_threshold,
            retriever_contributions={},
            filtered_below_threshold=0
        )
    
    def analyze_retriever_overlap(
        self, 
        retriever_results: Dict[str, List[Document]]
    ) -> Dict[str, Any]:
        """
        Analyze overlap between retrievers for diagnostic purposes.
        
        Args:
            retriever_results: Dict mapping retriever name to its document results
            
        Returns:
            Dictionary with overlap analysis
        """
        if len(retriever_results) < 2:
            return {"error": "Need at least 2 retrievers to analyze overlap"}
        
        # Get document IDs for each retriever
        retriever_doc_ids = {}
        for name, docs in retriever_results.items():
            retriever_doc_ids[name] = {self._get_document_id(doc) for doc in docs}
        
        # Calculate pairwise overlaps
        overlaps = {}
        retriever_names = list(retriever_doc_ids.keys())
        
        for i, ret1 in enumerate(retriever_names):
            for ret2 in retriever_names[i+1:]:
                set1 = retriever_doc_ids[ret1]
                set2 = retriever_doc_ids[ret2]
                
                intersection = set1 & set2
                union = set1 | set2
                
                overlap_pct = len(intersection) / len(union) * 100 if union else 0
                
                overlaps[f"{ret1}_vs_{ret2}"] = {
                    "intersection_size": len(intersection),
                    "union_size": len(union),
                    "overlap_percentage": overlap_pct
                }
        
        # Overall statistics
        all_doc_ids = set()
        for doc_ids in retriever_doc_ids.values():
            all_doc_ids.update(doc_ids)
        
        total_docs_before = sum(len(docs) for docs in retriever_results.values())
        unique_docs_after = len(all_doc_ids)
        dedup_ratio = (total_docs_before - unique_docs_after) / total_docs_before * 100 if total_docs_before else 0
        
        return {
            "pairwise_overlaps": overlaps,
            "total_docs_before_merge": total_docs_before,
            "unique_docs_after_merge": unique_docs_after,
            "deduplication_ratio_pct": dedup_ratio,
            "retriever_sizes": {name: len(docs) for name, docs in retriever_results.items()}
        }


def create_rrf_merger(
    k: int = 60,
    normalize_scores: bool = True,
    score_threshold: float = 0.0
) -> RRFMerger:
    """
    Factory function to create an RRF merger with recommended settings.
    
    Args:
        k: RRF parameter (60 recommended for balanced recall/precision)
        normalize_scores: Whether to normalize final scores
        score_threshold: Minimum normalized score to keep documents
        
    Returns:
        Configured RRFMerger instance
    """
    return RRFMerger(
        k=k,
        normalize_scores=normalize_scores,
        score_threshold=score_threshold
    )
