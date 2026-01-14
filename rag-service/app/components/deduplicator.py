"""
Document deduplication pipeline using exact ID matching and near-duplicate detection.

Implements a two-stage deduplication process:
1. Stage 1: Exact document ID matching
2. Stage 2: Near-duplicate detection using MinHash (Jaccard similarity) and SimHash (Hamming distance)

This component ensures document uniqueness while preserving recall by avoiding
over-aggressive deduplication across document versions.
"""

import time
import hashlib
import re
from typing import List, Dict, Set, Tuple, Optional, Any
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from langchain_core.documents import Document
from datasketch import MinHashLSH, MinHash
from simhash import Simhash

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DeduplicationStats:
    """Statistics from deduplication operation."""
    total_docs_input: int
    unique_docs_output: int
    duplicates_removed: int
    stage1_exact_duplicates: int
    stage2_near_duplicates: int
    processing_time_ms: float
    
    # Breakdown by deduplication method
    id_duplicates: int = 0
    minhash_duplicates: int = 0
    simhash_duplicates: int = 0
    
    # Version preservation stats
    versions_preserved: int = 0
    
    def get_deduplication_ratio(self) -> float:
        """Calculate deduplication ratio as percentage."""
        if self.total_docs_input == 0:
            return 0.0
        return (self.duplicates_removed / self.total_docs_input) * 100


@dataclass
class DedupDocument:
    """Document with deduplication metadata."""
    document: Document
    original_index: int
    dedup_signature: str
    is_duplicate: bool = False
    duplicate_of: Optional[str] = None
    similarity_score: float = 0.0
    
    @property
    def id(self) -> str:
        """Get document ID."""
        return self.document.metadata.get("id", f"doc_{self.original_index}")


class DocumentDeduplicator:
    """
    Two-stage document deduplication pipeline.
    
    Stage 1: Exact ID matching - removes documents with identical IDs
    Stage 2: Near-duplicate detection using MinHash and SimHash
    
    Features:
    - Configurable similarity thresholds
    - Document version preservation
    - Comprehensive statistics tracking
    - Content preprocessing for better matching
    """
    
    def __init__(
        self,
        jaccard_threshold: float = 0.82,
        hamming_threshold: int = 4,
        minhash_num_perm: int = 128,
        content_window: int = 512,
        preserve_versions: bool = True
    ):
        """
        Initialize deduplicator with configurable parameters.
        
        Args:
            jaccard_threshold: MinHash Jaccard similarity threshold (≥0.82 recommended)
            hamming_threshold: SimHash Hamming distance threshold (≤4 recommended)  
            minhash_num_perm: Number of permutations for MinHash (128 for good precision)
            content_window: Number of characters to use for similarity (512 recommended)
            preserve_versions: Never deduplicate across document versions
        """
        self.jaccard_threshold = jaccard_threshold
        self.hamming_threshold = hamming_threshold
        self.minhash_num_perm = minhash_num_perm
        self.content_window = content_window
        self.preserve_versions = preserve_versions
        
        # Initialize LSH for efficient MinHash similarity search
        self.lsh = MinHashLSH(threshold=jaccard_threshold, num_perm=minhash_num_perm)
        
        # Validation
        if not (0.5 <= jaccard_threshold <= 1.0):
            raise ValueError(f"Jaccard threshold {jaccard_threshold} must be between 0.5 and 1.0")
        if not (1 <= hamming_threshold <= 10):
            raise ValueError(f"Hamming threshold {hamming_threshold} must be between 1 and 10")
        
        logger.info(f"Deduplicator initialized: Jaccard≥{jaccard_threshold}, "
                   f"Hamming≤{hamming_threshold}, content_window={content_window}")
    
    def deduplicate(self, documents: List[Document]) -> Tuple[List[Document], DeduplicationStats]:
        """
        Remove duplicate documents using two-stage process.
        
        Args:
            documents: List of documents to deduplicate
            
        Returns:
            Tuple of (unique_documents, deduplication_statistics)
        """
        start_time = time.time()
        
        if not documents:
            return [], self._create_empty_stats()
        
        logger.info(f"Starting deduplication of {len(documents)} documents")
        
        # Convert to DedupDocument objects with metadata
        dedup_docs = [
            DedupDocument(
                document=doc,
                original_index=i,
                dedup_signature=self._generate_signature(doc)
            )
            for i, doc in enumerate(documents)
        ]
        
        # Stage 1: Exact ID deduplication
        stage1_unique = self._stage1_exact_deduplication(dedup_docs)
        stage1_removed = len(dedup_docs) - len(stage1_unique)
        
        # Stage 2: Near-duplicate detection
        stage2_unique = self._stage2_near_duplicate_detection(stage1_unique)
        stage2_removed = len(stage1_unique) - len(stage2_unique)
        
        # Extract final documents
        unique_documents = [dd.document for dd in stage2_unique]
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Create statistics
        stats = DeduplicationStats(
            total_docs_input=len(documents),
            unique_docs_output=len(unique_documents),
            duplicates_removed=len(documents) - len(unique_documents),
            stage1_exact_duplicates=stage1_removed,
            stage2_near_duplicates=stage2_removed,
            processing_time_ms=processing_time_ms
        )
        
        logger.info(f"Deduplication completed: {len(documents)} → {len(unique_documents)} "
                   f"({stats.get_deduplication_ratio():.1f}% removed) in {processing_time_ms:.2f}ms")
        
        return unique_documents, stats
    
    def _stage1_exact_deduplication(self, dedup_docs: List[DedupDocument]) -> List[DedupDocument]:
        """
        Stage 1: Remove documents with exact ID matches.
        
        Args:
            dedup_docs: List of documents to process
            
        Returns:
            List of documents with exact duplicates removed
        """
        seen_ids: Set[str] = set()
        version_groups: Dict[str, List[DedupDocument]] = defaultdict(list)
        unique_docs = []
        
        for doc in dedup_docs:
            doc_id = self._extract_id(doc.document)
            
            # Handle version preservation
            if self.preserve_versions:
                # Extract base ID without version
                base_id = self._extract_base_id(doc_id)
                version = self._extract_version(doc.document)
                
                # Group by base ID and version
                version_key = f"{base_id}_v{version}"
                version_groups[version_key].append(doc)
            else:
                # Simple ID-based deduplication
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    unique_docs.append(doc)
        
        # Process version groups if version preservation is enabled
        if self.preserve_versions:
            for version_key, docs in version_groups.items():
                if docs:
                    # Keep first document in each version group
                    unique_docs.append(docs[0])
                    # Mark others as duplicates
                    for dup_doc in docs[1:]:
                        dup_doc.is_duplicate = True
                        dup_doc.duplicate_of = docs[0].id
        
        stage1_unique = [doc for doc in unique_docs if not doc.is_duplicate]
        logger.debug(f"Stage 1: {len(dedup_docs)} → {len(stage1_unique)} "
                    f"({len(dedup_docs) - len(stage1_unique)} exact duplicates removed)")
        
        return stage1_unique
    
    def _stage2_near_duplicate_detection(self, dedup_docs: List[DedupDocument]) -> List[DedupDocument]:
        """
        Stage 2: Near-duplicate detection using MinHash and SimHash.
        
        Args:
            dedup_docs: List of documents from stage 1
            
        Returns:
            List of documents with near-duplicates removed
        """
        if len(dedup_docs) <= 1:
            return dedup_docs
        
        # Clear LSH from any previous runs
        self.lsh = MinHashLSH(threshold=self.jaccard_threshold, num_perm=self.minhash_num_perm)
        
        # Generate MinHash and SimHash signatures
        minhash_sigs = {}
        simhash_sigs = {}
        
        for doc in dedup_docs:
            content = self._prepare_content_for_hashing(doc.document)
            
            # MinHash signature
            minhash = self._generate_minhash(content)
            minhash_sigs[doc.id] = minhash
            
            # SimHash signature
            simhash = self._generate_simhash(content)
            simhash_sigs[doc.id] = simhash
            
            # Add to LSH for efficient similarity search
            try:
                self.lsh.insert(doc.id, minhash)
            except ValueError:
                # Document might be too similar to existing ones
                logger.debug(f"Could not insert document {doc.id} into LSH")
        
        # Find near-duplicates
        unique_docs = []
        duplicate_ids = set()
        
        for doc in dedup_docs:
            if doc.id in duplicate_ids:
                continue
            
            # Find MinHash candidates
            minhash_candidates = self.lsh.query(minhash_sigs[doc.id])
            minhash_candidates = [cid for cid in minhash_candidates if cid != doc.id and cid not in duplicate_ids]
            
            # Find SimHash candidates
            simhash_candidates = []
            for other_doc in dedup_docs:
                if (other_doc.id != doc.id and 
                    other_doc.id not in duplicate_ids and
                    self._simhash_distance(simhash_sigs[doc.id], simhash_sigs[other_doc.id]) <= self.hamming_threshold):
                    simhash_candidates.append(other_doc.id)
            
            # Mark duplicates from both methods
            all_candidates = set(minhash_candidates + simhash_candidates)
            
            if all_candidates:
                # Check version preservation
                final_duplicates = []
                for candidate_id in all_candidates:
                    candidate_doc = next(d for d in dedup_docs if d.id == candidate_id)
                    if not self._should_preserve_as_different_version(doc.document, candidate_doc.document):
                        final_duplicates.append(candidate_id)
                
                # Mark as duplicates
                duplicate_ids.update(final_duplicates)
                
                logger.debug(f"Document {doc.id} has {len(final_duplicates)} near-duplicates")
            
            unique_docs.append(doc)
        
        logger.debug(f"Stage 2: {len(dedup_docs)} → {len(unique_docs)} "
                    f"({len(duplicate_ids)} near-duplicates removed)")
        
        return unique_docs
    
    def _generate_signature(self, doc: Document) -> str:
        """Generate a unique signature for a document."""
        content = self._prepare_content_for_hashing(doc)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _prepare_content_for_hashing(self, doc: Document) -> str:
        """
        Prepare document content for hashing.
        
        Uses title + first N characters of content for similarity comparison.
        
        Args:
            doc: Document to prepare
            
        Returns:
            Processed content string
        """
        # Extract title from metadata
        title = doc.metadata.get("title", "")
        if not title:
            # Try other title fields
            title = doc.metadata.get("heading", "") or doc.metadata.get("section", "")
        
        # Get content window
        content = doc.page_content[:self.content_window]
        
        # Combine title and content
        combined = f"{title} {content}".strip()
        
        # Normalize text: lowercase, remove extra whitespace, normalize punctuation
        normalized = re.sub(r'\s+', ' ', combined.lower())
        normalized = re.sub(r'[^\w\s]', ' ', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def _generate_minhash(self, content: str) -> MinHash:
        """Generate MinHash signature for content."""
        minhash = MinHash(num_perm=self.minhash_num_perm)
        
        # Create shingles (3-grams of words)
        words = content.split()
        for i in range(max(1, len(words) - 2)):
            shingle = ' '.join(words[i:i+3])
            minhash.update(shingle.encode('utf8'))
        
        return minhash
    
    def _generate_simhash(self, content: str) -> Simhash:
        """Generate SimHash signature for content."""
        return Simhash(content)
    
    def _simhash_distance(self, simhash1: Simhash, simhash2: Simhash) -> int:
        """Calculate Hamming distance between two SimHash signatures."""
        return simhash1.distance(simhash2)
    
    def _extract_id(self, doc: Document) -> str:
        """Extract document ID from metadata."""
        for id_field in ['id', 'doc_id', 'document_id', 'source_id']:
            if id_field in doc.metadata and doc.metadata[id_field]:
                return str(doc.metadata[id_field])
        
        # Generate hash-based ID if no explicit ID
        return f"doc_{hash(doc.page_content)}"
    
    def _extract_base_id(self, doc_id: str) -> str:
        """Extract base ID without version information."""
        # Remove common version patterns
        base_id = re.sub(r'_v\d+$', '', doc_id)  # Remove _v1, _v2, etc.
        base_id = re.sub(r'_\d{4}-\d{2}-\d{2}$', '', base_id)  # Remove date suffixes
        base_id = re.sub(r'_rev\d+$', '', base_id)  # Remove revision numbers
        return base_id
    
    def _extract_version(self, doc: Document) -> str:
        """Extract version information from document metadata."""
        # Try explicit version fields
        for version_field in ['version', 'revision', 'rev', 'v']:
            if version_field in doc.metadata:
                return str(doc.metadata[version_field])
        
        # Extract from ID patterns
        doc_id = self._extract_id(doc)
        version_match = re.search(r'_v(\d+)$', doc_id)
        if version_match:
            return version_match.group(1)
        
        # Extract from date patterns
        date_match = re.search(r'_(\d{4}-\d{2}-\d{2})$', doc_id)
        if date_match:
            return date_match.group(1)
        
        return "1"  # Default version
    
    def _should_preserve_as_different_version(self, doc1: Document, doc2: Document) -> bool:
        """
        Check if two documents should be preserved as different versions.
        
        Args:
            doc1: First document
            doc2: Second document
            
        Returns:
            True if documents should be preserved as different versions
        """
        if not self.preserve_versions:
            return False
        
        # Extract base IDs and versions
        id1 = self._extract_id(doc1)
        id2 = self._extract_id(doc2)
        
        base_id1 = self._extract_base_id(id1)
        base_id2 = self._extract_base_id(id2)
        
        # If base IDs are different, not versions of same document
        if base_id1 != base_id2:
            return False
        
        # If base IDs are same, check versions
        version1 = self._extract_version(doc1)
        version2 = self._extract_version(doc2)
        
        # Preserve if versions are different
        return version1 != version2
    
    def _create_empty_stats(self) -> DeduplicationStats:
        """Create empty statistics object."""
        return DeduplicationStats(
            total_docs_input=0,
            unique_docs_output=0,
            duplicates_removed=0,
            stage1_exact_duplicates=0,
            stage2_near_duplicates=0,
            processing_time_ms=0.0
        )
    
    def analyze_duplicates(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Analyze duplicate patterns in document collection.
        
        Args:
            documents: List of documents to analyze
            
        Returns:
            Analysis results with duplicate patterns and statistics
        """
        if not documents:
            return {"error": "No documents to analyze"}
        
        # Group by exact IDs
        id_groups = defaultdict(list)
        for i, doc in enumerate(documents):
            doc_id = self._extract_id(doc)
            id_groups[doc_id].append(i)
        
        # Find exact duplicates
        exact_duplicates = {doc_id: indices for doc_id, indices in id_groups.items() if len(indices) > 1}
        
        # Find near-duplicates using similarity
        near_duplicates = []
        processed_pairs = set()
        
        for i, doc1 in enumerate(documents):
            for j, doc2 in enumerate(documents[i+1:], i+1):
                if (i, j) in processed_pairs:
                    continue
                
                content1 = self._prepare_content_for_hashing(doc1)
                content2 = self._prepare_content_for_hashing(doc2)
                
                # Calculate MinHash similarity
                mh1 = self._generate_minhash(content1)
                mh2 = self._generate_minhash(content2)
                jaccard_sim = mh1.jaccard(mh2)
                
                # Calculate SimHash distance
                sh1 = self._generate_simhash(content1)
                sh2 = self._generate_simhash(content2)
                hamming_dist = sh1.distance(sh2)
                
                if jaccard_sim >= self.jaccard_threshold or hamming_dist <= self.hamming_threshold:
                    near_duplicates.append({
                        "doc1_index": i,
                        "doc2_index": j,
                        "jaccard_similarity": jaccard_sim,
                        "hamming_distance": hamming_dist,
                        "doc1_id": self._extract_id(doc1),
                        "doc2_id": self._extract_id(doc2)
                    })
                
                processed_pairs.add((i, j))
        
        return {
            "total_documents": len(documents),
            "unique_ids": len(id_groups),
            "exact_duplicate_groups": len(exact_duplicates),
            "exact_duplicate_count": sum(len(indices) - 1 for indices in exact_duplicates.values()),
            "near_duplicate_pairs": len(near_duplicates),
            "exact_duplicates": dict(exact_duplicates),
            "near_duplicates": near_duplicates,
            "estimated_deduplication_ratio": (
                sum(len(indices) - 1 for indices in exact_duplicates.values()) + len(near_duplicates)
            ) / len(documents) * 100
        }


def create_deduplicator(
    jaccard_threshold: float = 0.82,
    hamming_threshold: int = 4,
    preserve_versions: bool = True
) -> DocumentDeduplicator:
    """
    Factory function to create a document deduplicator with recommended settings.
    
    Args:
        jaccard_threshold: MinHash Jaccard similarity threshold (≥0.82 for recall preservation)
        hamming_threshold: SimHash Hamming distance threshold (≤4 for precision)
        preserve_versions: Never deduplicate across document versions
        
    Returns:
        Configured DocumentDeduplicator instance
    """
    return DocumentDeduplicator(
        jaccard_threshold=jaccard_threshold,
        hamming_threshold=hamming_threshold,
        preserve_versions=preserve_versions
    )