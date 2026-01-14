"""LSH-based deduplication for efficient near-duplicate detection.

Uses MinHash LSH for O(1) approximate nearest neighbor pre-screening,
reducing the need for expensive TF-IDF similarity computations.
"""

import hashlib
import re
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
from datasketch import MinHash, MinHashLSH

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LSHConfig:
    """Configuration for LSH deduplication."""
    num_perm: int = 128  # Number of permutations for MinHash
    threshold: float = 0.5  # LSH similarity threshold
    weights: Tuple[float, float] = (0.5, 0.5)  # (false_positive, false_negative) weights
    redis_enabled: bool = True  # Use Redis for persistence
    redis_key_prefix: str = "lsh:dedup:"
    shingle_size: int = 3  # Word n-gram size for shingling


class LSHIndex:
    """MinHash LSH index for efficient near-duplicate detection.

    This index enables O(1) approximate nearest neighbor lookups,
    dramatically reducing the cost of deduplication for large corpora.
    """

    def __init__(
        self,
        config: Optional[LSHConfig] = None,
        cache_service: Optional[Any] = None
    ):
        """Initialize LSH index.

        Args:
            config: LSH configuration
            cache_service: Redis cache service for persistence
        """
        self.config = config or LSHConfig()
        self.cache_service = cache_service

        # Initialize LSH index
        self.lsh = MinHashLSH(
            threshold=self.config.threshold,
            num_perm=self.config.num_perm,
            weights=self.config.weights
        )

        # In-memory signature cache for batch operations
        self._signature_cache: Dict[str, MinHash] = {}

        # Track indexed document IDs
        self._indexed_ids: Set[str] = set()

        # Metrics
        self._metrics = {
            "queries": 0,
            "candidates_found": 0,
            "false_positives_filtered": 0,
        }

    def _normalize_content(self, content: str) -> str:
        """Normalize content for consistent hashing."""
        # Remove extra whitespace
        content = re.sub(r'\s+', ' ', content)
        # Remove punctuation variations
        content = re.sub(r'[""''„"«»]', '"', content)
        content = re.sub(r'[–—]', '-', content)
        # Lowercase
        content = content.lower().strip()
        return content

    def _get_shingles(self, content: str) -> Set[str]:
        """Generate word n-gram shingles from content."""
        normalized = self._normalize_content(content)
        words = normalized.split()

        if len(words) < self.config.shingle_size:
            # For short content, use the whole thing
            return {normalized}

        shingles = set()
        for i in range(len(words) - self.config.shingle_size + 1):
            shingle = ' '.join(words[i:i + self.config.shingle_size])
            shingles.add(shingle)

        return shingles

    def _compute_minhash(self, content: str) -> MinHash:
        """Compute MinHash signature for content."""
        minhash = MinHash(num_perm=self.config.num_perm)

        shingles = self._get_shingles(content)
        for shingle in shingles:
            minhash.update(shingle.encode('utf-8'))

        return minhash

    def _get_content_id(self, content: str) -> str:
        """Generate a unique ID for content."""
        normalized = self._normalize_content(content)
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:16]

    def add(self, doc_id: str, content: str) -> bool:
        """Add a document to the LSH index.

        Args:
            doc_id: Unique document identifier
            content: Document content

        Returns:
            True if added successfully, False if already indexed
        """
        if doc_id in self._indexed_ids:
            logger.debug(f"Document {doc_id} already indexed")
            return False

        try:
            minhash = self._compute_minhash(content)

            # Add to LSH index
            self.lsh.insert(doc_id, minhash)

            # Cache signature for potential updates
            self._signature_cache[doc_id] = minhash
            self._indexed_ids.add(doc_id)

            # Persist to Redis if enabled
            if self.cache_service and self.config.redis_enabled:
                self._persist_signature(doc_id, minhash)

            return True

        except Exception as e:
            logger.error(f"Failed to add document {doc_id} to LSH index: {e}")
            return False

    def add_batch(self, documents: List[Dict[str, Any]], id_key: str = "id", content_key: str = "content") -> int:
        """Add multiple documents to the LSH index.

        Args:
            documents: List of documents with id and content
            id_key: Key for document ID
            content_key: Key for document content

        Returns:
            Number of documents successfully added
        """
        added = 0
        for doc in documents:
            doc_id = doc.get(id_key)
            content = doc.get(content_key)

            if doc_id and content:
                if self.add(doc_id, content):
                    added += 1

        logger.info(f"Added {added}/{len(documents)} documents to LSH index")
        return added

    def query(self, content: str) -> List[str]:
        """Find candidate duplicates for content.

        Args:
            content: Content to check for duplicates

        Returns:
            List of candidate document IDs that may be duplicates
        """
        self._metrics["queries"] += 1

        try:
            minhash = self._compute_minhash(content)
            candidates = self.lsh.query(minhash)

            self._metrics["candidates_found"] += len(candidates)

            return list(candidates)

        except Exception as e:
            logger.error(f"LSH query failed: {e}")
            return []

    def query_batch(self, contents: List[str]) -> Dict[int, List[str]]:
        """Query multiple contents for duplicates.

        Args:
            contents: List of contents to check

        Returns:
            Dict mapping content index to list of candidate IDs
        """
        results = {}
        for idx, content in enumerate(contents):
            candidates = self.query(content)
            if candidates:
                results[idx] = candidates
        return results

    def remove(self, doc_id: str) -> bool:
        """Remove a document from the LSH index.

        Args:
            doc_id: Document ID to remove

        Returns:
            True if removed successfully
        """
        if doc_id not in self._indexed_ids:
            return False

        try:
            self.lsh.remove(doc_id)
            self._indexed_ids.discard(doc_id)
            self._signature_cache.pop(doc_id, None)

            # Remove from Redis if enabled
            if self.cache_service and self.config.redis_enabled:
                self._remove_persisted_signature(doc_id)

            return True

        except Exception as e:
            logger.error(f"Failed to remove document {doc_id} from LSH index: {e}")
            return False

    def is_potential_duplicate(self, content: str, threshold: Optional[float] = None) -> Tuple[bool, List[str]]:
        """Quick check if content is a potential duplicate.

        Args:
            content: Content to check
            threshold: Optional override for similarity threshold

        Returns:
            Tuple of (is_potential_duplicate, candidate_ids)
        """
        candidates = self.query(content)
        return len(candidates) > 0, candidates

    def estimate_similarity(self, content1: str, content2: str) -> float:
        """Estimate Jaccard similarity using MinHash.

        This is faster than exact similarity but approximate.

        Args:
            content1: First content
            content2: Second content

        Returns:
            Estimated Jaccard similarity (0-1)
        """
        minhash1 = self._compute_minhash(content1)
        minhash2 = self._compute_minhash(content2)
        return minhash1.jaccard(minhash2)

    def _persist_signature(self, doc_id: str, minhash: MinHash):
        """Persist MinHash signature to Redis."""
        if not self.cache_service:
            return

        try:
            key = f"{self.config.redis_key_prefix}{doc_id}"
            # Serialize MinHash hashvalues
            import json
            data = json.dumps(minhash.hashvalues.tolist())
            # Use async in sync context - this will be called from async code
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self.cache_service.set(key, data, ttl=604800))  # 1 week
                else:
                    loop.run_until_complete(self.cache_service.set(key, data, ttl=604800))
            except RuntimeError:
                # No event loop, skip persistence
                pass
        except Exception as e:
            logger.warning(f"Failed to persist LSH signature for {doc_id}: {e}")

    def _remove_persisted_signature(self, doc_id: str):
        """Remove persisted MinHash signature from Redis."""
        if not self.cache_service:
            return

        try:
            key = f"{self.config.redis_key_prefix}{doc_id}"
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self.cache_service.delete(key))
                else:
                    loop.run_until_complete(self.cache_service.delete(key))
            except RuntimeError:
                pass
        except Exception as e:
            logger.warning(f"Failed to remove LSH signature for {doc_id}: {e}")

    async def load_from_redis(self) -> int:
        """Load all persisted signatures from Redis.

        Returns:
            Number of signatures loaded
        """
        if not self.cache_service or not self.config.redis_enabled:
            return 0

        loaded = 0
        try:
            # This would require a scan operation - implementation depends on cache service
            logger.info("LSH index load from Redis - requires cache service scan implementation")
        except Exception as e:
            logger.error(f"Failed to load LSH signatures from Redis: {e}")

        return loaded

    def get_metrics(self) -> Dict[str, Any]:
        """Get LSH index metrics."""
        return {
            "indexed_documents": len(self._indexed_ids),
            "cached_signatures": len(self._signature_cache),
            "queries": self._metrics["queries"],
            "candidates_found": self._metrics["candidates_found"],
            "avg_candidates_per_query": (
                self._metrics["candidates_found"] / max(1, self._metrics["queries"])
            ),
            "config": {
                "num_perm": self.config.num_perm,
                "threshold": self.config.threshold,
                "shingle_size": self.config.shingle_size,
            }
        }

    def clear(self):
        """Clear the LSH index."""
        self.lsh = MinHashLSH(
            threshold=self.config.threshold,
            num_perm=self.config.num_perm,
            weights=self.config.weights
        )
        self._signature_cache.clear()
        self._indexed_ids.clear()
        self._metrics = {
            "queries": 0,
            "candidates_found": 0,
            "false_positives_filtered": 0,
        }
        logger.info("LSH index cleared")


class LSHDeduplicationFilter:
    """High-level deduplication filter using LSH pre-screening.

    This class provides the main interface for deduplication,
    combining LSH pre-screening with exact similarity verification.
    """

    def __init__(
        self,
        lsh_index: Optional[LSHIndex] = None,
        similarity_threshold: float = 0.85,
        config: Optional[LSHConfig] = None,
        cache_service: Optional[Any] = None
    ):
        """Initialize deduplication filter.

        Args:
            lsh_index: Existing LSH index or None to create new
            similarity_threshold: Threshold for exact similarity check
            config: LSH configuration
            cache_service: Redis cache service
        """
        self.lsh_index = lsh_index or LSHIndex(config=config, cache_service=cache_service)
        self.similarity_threshold = similarity_threshold

        # Metrics
        self._metrics = {
            "total_checks": 0,
            "lsh_filtered": 0,  # Passed LSH but failed exact check
            "duplicates_found": 0,
            "unique_documents": 0,
        }

    def check_duplicate(
        self,
        content: str,
        get_content_func: Optional[callable] = None
    ) -> Tuple[bool, Optional[str], float]:
        """Check if content is a duplicate of existing documents.

        Args:
            content: Content to check
            get_content_func: Function to retrieve content by doc_id for exact check

        Returns:
            Tuple of (is_duplicate, matching_doc_id, similarity_score)
        """
        self._metrics["total_checks"] += 1

        # LSH pre-screening
        is_potential, candidates = self.lsh_index.is_potential_duplicate(content)

        if not is_potential:
            self._metrics["unique_documents"] += 1
            return False, None, 0.0

        # If we have candidates and a way to get their content, do exact check
        if get_content_func:
            for candidate_id in candidates:
                try:
                    candidate_content = get_content_func(candidate_id)
                    if candidate_content:
                        similarity = self.lsh_index.estimate_similarity(content, candidate_content)
                        if similarity >= self.similarity_threshold:
                            self._metrics["duplicates_found"] += 1
                            return True, candidate_id, similarity
                except Exception as e:
                    logger.warning(f"Failed to retrieve content for {candidate_id}: {e}")

            # Had candidates but none passed exact check
            self._metrics["lsh_filtered"] += 1
            self._metrics["unique_documents"] += 1
            return False, None, 0.0

        # No exact check available, trust LSH result
        self._metrics["duplicates_found"] += 1
        return True, candidates[0] if candidates else None, self.lsh_index.config.threshold

    def filter_duplicates(
        self,
        documents: List[Dict[str, Any]],
        id_key: str = "id",
        content_key: str = "content"
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Filter duplicates from a list of documents.

        Args:
            documents: Documents to filter
            id_key: Key for document ID
            content_key: Key for document content

        Returns:
            Tuple of (unique_documents, duplicate_documents)
        """
        unique = []
        duplicates = []

        # Build temporary index for within-batch dedup
        temp_index = LSHIndex(config=self.lsh_index.config)

        for doc in documents:
            doc_id = doc.get(id_key)
            content = doc.get(content_key)

            if not doc_id or not content:
                continue

            # Check against existing index
            is_dup, match_id, score = self.check_duplicate(content)

            if is_dup:
                doc["duplicate_of"] = match_id
                doc["similarity_score"] = score
                duplicates.append(doc)
                continue

            # Check against batch-local index
            local_candidates = temp_index.query(content)
            if local_candidates:
                # Found duplicate within batch
                doc["duplicate_of"] = local_candidates[0]
                doc["similarity_score"] = temp_index.estimate_similarity(
                    content,
                    next((d.get(content_key) for d in unique if d.get(id_key) == local_candidates[0]), "")
                )
                duplicates.append(doc)
                continue

            # Unique document
            unique.append(doc)
            temp_index.add(doc_id, content)

        logger.info(f"Deduplication: {len(unique)} unique, {len(duplicates)} duplicates from {len(documents)} documents")

        return unique, duplicates

    def get_metrics(self) -> Dict[str, Any]:
        """Get deduplication filter metrics."""
        return {
            **self._metrics,
            "lsh_index": self.lsh_index.get_metrics(),
            "filter_rate": (
                self._metrics["lsh_filtered"] / max(1, self._metrics["total_checks"])
            ),
            "duplicate_rate": (
                self._metrics["duplicates_found"] / max(1, self._metrics["total_checks"])
            ),
        }
