"""Deduplication utilities for document ingestion."""

import hashlib
import re
from typing import List, Dict, Any, Optional, Tuple
from difflib import SequenceMatcher
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.core.logging import get_logger
from app.core.config import settings

logger = get_logger(__name__)

# Lazy import for LSH to avoid circular imports
_lsh_index = None
_lsh_filter = None


def get_lsh_index():
    """Get or create the global LSH index instance."""
    global _lsh_index
    if _lsh_index is None and settings.dedup_lsh_enabled:
        from app.utils.lsh_dedup import LSHIndex, LSHConfig
        config = LSHConfig(
            num_perm=settings.dedup_lsh_num_perm,
            threshold=settings.dedup_lsh_threshold,
            shingle_size=settings.dedup_lsh_shingle_size,
            redis_enabled=settings.dedup_lsh_redis_persist,
        )
        _lsh_index = LSHIndex(config=config)
        logger.info("LSH index initialized for deduplication pre-screening")
    return _lsh_index


def get_lsh_filter():
    """Get or create the global LSH deduplication filter."""
    global _lsh_filter
    if _lsh_filter is None and settings.dedup_lsh_enabled:
        from app.utils.lsh_dedup import LSHDeduplicationFilter
        _lsh_filter = LSHDeduplicationFilter(
            lsh_index=get_lsh_index(),
            similarity_threshold=settings.dedup_jaccard_threshold,
        )
        logger.info("LSH deduplication filter initialized")
    return _lsh_filter


class ContentHasher:
    """Generates various hashes for content deduplication."""
    
    @staticmethod
    def generate_content_hash(content: str, algorithm: str = "sha256") -> str:
        """Generate hash of content."""
        # Normalize content for consistent hashing
        normalized = ContentHasher._normalize_content(content)
        
        if algorithm == "md5":
            hasher = hashlib.md5()
        elif algorithm == "sha1":
            hasher = hashlib.sha1()
        elif algorithm == "sha256":
            hasher = hashlib.sha256()
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
            
        hasher.update(normalized.encode('utf-8'))
        return hasher.hexdigest()
        
    @staticmethod
    def generate_fuzzy_hash(content: str, n_grams: int = 5) -> str:
        """Generate fuzzy hash using n-grams for near-duplicate detection."""
        normalized = ContentHasher._normalize_content(content)
        
        # Extract n-grams
        ngrams = []
        for i in range(len(normalized) - n_grams + 1):
            ngrams.append(normalized[i:i + n_grams])
            
        # Sort and hash n-grams
        ngrams = sorted(set(ngrams))
        
        # Create hash from top n-grams
        hasher = hashlib.md5()
        for ngram in ngrams[:100]:  # Use top 100 n-grams
            hasher.update(ngram.encode('utf-8'))
            
        return hasher.hexdigest()
        
    @staticmethod
    def generate_semantic_hash(content: str, dimensions: int = 8) -> str:
        """Generate semantic hash using dimensionality reduction."""
        normalized = ContentHasher._normalize_content(content)
        
        # Simple semantic hash using character frequencies
        char_freq = {}
        for char in normalized.lower():
            if char.isalnum():
                char_freq[char] = char_freq.get(char, 0) + 1
                
        # Create vector from frequencies
        vector = []
        for char in 'abcdefghijklmnopqrstuvwxyz0123456789':
            vector.append(char_freq.get(char, 0))
            
        # Reduce to specified dimensions
        if len(vector) > dimensions:
            # Simple dimensionality reduction
            reduced = []
            chunk_size = len(vector) // dimensions
            for i in range(0, len(vector), chunk_size):
                chunk = vector[i:i + chunk_size]
                reduced.append(sum(chunk))
            vector = reduced[:dimensions]
            
        # Convert to hash
        hasher = hashlib.md5()
        for val in vector:
            hasher.update(str(val).encode('utf-8'))
            
        return hasher.hexdigest()
        
    @staticmethod
    def _normalize_content(content: str) -> str:
        """Normalize content for consistent hashing."""
        # Remove extra whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Remove punctuation variations
        content = re.sub(r'[""''„"«»]', '"', content)
        content = re.sub(r'[–—]', '-', content)
        
        # Lowercase
        content = content.lower()
        
        # Strip leading/trailing whitespace
        content = content.strip()
        
        return content


class DeduplicationService:
    """Service for detecting and handling duplicate documents."""

    def __init__(self, similarity_threshold: float = 0.85, use_lsh: bool = True):
        """Initialize deduplication service.

        Args:
            similarity_threshold: Threshold for exact similarity matching
            use_lsh: Whether to use LSH pre-screening (default: True if enabled in config)
        """
        self.similarity_threshold = similarity_threshold
        self.content_hasher = ContentHasher()
        self._vectorizer = None
        self._use_lsh = use_lsh and settings.dedup_lsh_enabled
        self._lsh_index = get_lsh_index() if self._use_lsh else None

        # Metrics for LSH effectiveness
        self._metrics = {
            "total_checks": 0,
            "lsh_skipped": 0,  # Checks skipped due to LSH pre-filter
            "exact_checks": 0,  # Full similarity checks performed
        }
        
    def calculate_similarity(self, content1: str, content2: str) -> float:
        """Calculate similarity between two pieces of content."""
        # Quick exact match check
        if content1 == content2:
            return 1.0
            
        # Use multiple similarity metrics
        scores = []
        
        # 1. Sequence matcher (good for similar structure)
        seq_score = SequenceMatcher(None, content1, content2).ratio()
        scores.append(seq_score)
        
        # 2. Jaccard similarity (set-based)
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        if words1 or words2:
            jaccard = len(words1.intersection(words2)) / len(words1.union(words2))
            scores.append(jaccard)
            
        # 3. TF-IDF cosine similarity (semantic)
        try:
            if not self._vectorizer:
                self._vectorizer = TfidfVectorizer(
                    max_features=1000,
                    ngram_range=(1, 2),
                    stop_words='english'
                )
                
            # Fit and transform both documents
            tfidf_matrix = self._vectorizer.fit_transform([content1, content2])
            cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            scores.append(cosine_sim)
            
        except Exception as e:
            logger.warning(f"TF-IDF similarity failed: {e}")
            
        # Return weighted average
        if scores:
            return sum(scores) / len(scores)
        return 0.0
        
    def is_duplicate(
        self,
        content: str,
        existing_content: str,
        metadata1: Optional[Dict[str, Any]] = None,
        metadata2: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None
    ) -> Tuple[bool, float, str]:
        """Check if content is duplicate of existing content.

        Uses LSH pre-screening when enabled to avoid expensive similarity calculations
        for content that is clearly different.

        Args:
            content: New content to check
            existing_content: Existing content to compare against
            metadata1: Metadata for new content
            metadata2: Metadata for existing content
            doc_id: Optional document ID for LSH indexing

        Returns:
            Tuple of (is_duplicate, similarity_score, reason)
        """
        self._metrics["total_checks"] += 1

        # Check exact content hash first (fastest check)
        hash1 = self.content_hasher.generate_content_hash(content)
        hash2 = self.content_hasher.generate_content_hash(existing_content)

        if hash1 == hash2:
            return True, 1.0, "exact_match"

        # LSH pre-screening: skip expensive similarity if LSH says not similar
        if self._use_lsh and self._lsh_index:
            lsh_similarity = self._lsh_index.estimate_similarity(content, existing_content)
            if lsh_similarity < settings.dedup_lsh_threshold:
                # LSH says not similar, skip expensive checks
                self._metrics["lsh_skipped"] += 1
                logger.debug(f"LSH pre-filter: skipped (estimated similarity: {lsh_similarity:.3f})")
                return False, lsh_similarity, "lsh_filtered"

        # Check fuzzy hash for near-duplicates
        fuzzy1 = self.content_hasher.generate_fuzzy_hash(content)
        fuzzy2 = self.content_hasher.generate_fuzzy_hash(existing_content)

        if fuzzy1 == fuzzy2:
            self._metrics["exact_checks"] += 1
            similarity = self.calculate_similarity(content, existing_content)
            if similarity >= self.similarity_threshold:
                return True, similarity, "fuzzy_match"

        # Calculate detailed similarity (expensive - TF-IDF)
        self._metrics["exact_checks"] += 1
        similarity = self.calculate_similarity(content, existing_content)

        if similarity >= self.similarity_threshold:
            return True, similarity, "high_similarity"

        # Check metadata for same source
        if metadata1 and metadata2:
            source1 = metadata1.get("source", "")
            source2 = metadata2.get("source", "")

            if source1 and source1 == source2:
                # Same source, check if it's an update
                modified1 = metadata1.get("last_modified")
                modified2 = metadata2.get("last_modified")

                if modified1 and modified2 and modified1 != modified2:
                    return False, similarity, "updated_version"

        return False, similarity, "not_duplicate"

    def add_to_lsh_index(self, doc_id: str, content: str) -> bool:
        """Add a document to the LSH index for future comparisons.

        Args:
            doc_id: Unique document identifier
            content: Document content

        Returns:
            True if added successfully
        """
        if self._use_lsh and self._lsh_index:
            return self._lsh_index.add(doc_id, content)
        return False

    def get_dedup_metrics(self) -> Dict[str, Any]:
        """Get deduplication metrics including LSH effectiveness."""
        metrics = self._metrics.copy()
        if self._use_lsh and self._lsh_index:
            metrics["lsh_index"] = self._lsh_index.get_metrics()
            metrics["lsh_skip_rate"] = (
                self._metrics["lsh_skipped"] / max(1, self._metrics["total_checks"])
            )
        return metrics
        
    def find_duplicates(
        self,
        documents: List[Dict[str, Any]],
        content_key: str = "content",
        id_key: str = "id"
    ) -> List[List[str]]:
        """Find duplicate groups in a list of documents.
        
        Returns:
            List of duplicate groups (each group is a list of document IDs)
        """
        n = len(documents)
        if n < 2:
            return []
            
        # Build similarity matrix
        visited = set()
        duplicate_groups = []
        
        for i in range(n):
            if i in visited:
                continue
                
            doc1 = documents[i]
            content1 = doc1.get(content_key, "")
            
            current_group = [doc1.get(id_key)]
            
            for j in range(i + 1, n):
                if j in visited:
                    continue
                    
                doc2 = documents[j]
                content2 = doc2.get(content_key, "")
                
                is_dup, score, reason = self.is_duplicate(
                    content1, content2,
                    doc1.get("metadata"), doc2.get("metadata")
                )
                
                if is_dup:
                    current_group.append(doc2.get(id_key))
                    visited.add(j)
                    
            if len(current_group) > 1:
                duplicate_groups.append(current_group)
                visited.add(i)
                
        return duplicate_groups
        
    def deduplicate_chunks(
        self,
        chunks: List[Dict[str, Any]],
        strategy: str = "keep_first"
    ) -> List[Dict[str, Any]]:
        """Remove duplicate chunks based on strategy.
        
        Strategies:
        - keep_first: Keep the first occurrence
        - keep_best: Keep the chunk with most metadata
        - merge: Merge metadata from duplicates
        """
        if not chunks:
            return []
            
        # Find duplicate groups
        duplicate_groups = self.find_duplicates(chunks)
        
        # Create a mapping of which chunks to keep
        chunks_to_remove = set()
        
        for group in duplicate_groups:
            if strategy == "keep_first":
                # Remove all but the first
                for chunk_id in group[1:]:
                    chunks_to_remove.add(chunk_id)
                    
            elif strategy == "keep_best":
                # Find chunk with most complete metadata
                best_idx = 0
                best_score = 0
                
                for idx, chunk_id in enumerate(group):
                    chunk = next(c for c in chunks if c.get("id") == chunk_id)
                    metadata = chunk.get("metadata", {})
                    
                    # Score based on metadata completeness
                    score = len([v for v in metadata.values() if v is not None])
                    
                    if score > best_score:
                        best_score = score
                        best_idx = idx
                        
                # Remove all but the best
                for idx, chunk_id in enumerate(group):
                    if idx != best_idx:
                        chunks_to_remove.add(chunk_id)
                        
            elif strategy == "merge":
                # Keep first but merge metadata
                base_chunk_id = group[0]
                base_chunk = next(c for c in chunks if c.get("id") == base_chunk_id)
                
                # Merge metadata from all duplicates
                merged_metadata = base_chunk.get("metadata", {}).copy()
                
                for chunk_id in group[1:]:
                    chunk = next(c for c in chunks if c.get("id") == chunk_id)
                    metadata = chunk.get("metadata", {})
                    
                    # Merge non-None values
                    for key, value in metadata.items():
                        if value is not None and merged_metadata.get(key) is None:
                            merged_metadata[key] = value
                            
                    chunks_to_remove.add(chunk_id)
                    
                # Update base chunk metadata
                base_chunk["metadata"] = merged_metadata
                
        # Filter out removed chunks
        deduplicated = [
            chunk for chunk in chunks
            if chunk.get("id") not in chunks_to_remove
        ]
        
        logger.info(
            f"Deduplication complete: {len(chunks)} -> {len(deduplicated)} chunks "
            f"({len(chunks_to_remove)} removed)"
        )
        
        return deduplicated