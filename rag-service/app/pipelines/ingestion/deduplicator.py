"""Document deduplication for ingestion pipeline."""

import asyncio
from typing import Any, Dict, List, Optional, Tuple

from app.core.logging import get_logger
from app.models.documents import Document, DocumentIngestionRequest
from app.utils.deduplication import DeduplicationService, ContentHasher

logger = get_logger(__name__)


class Deduplicator:
    """Deduplicates documents during ingestion."""

    def __init__(
        self,
        threshold: float = 0.85,
        vector_store_manager: Optional[Any] = None,
    ):
        """Initialize deduplicator.

        Args:
            threshold: Similarity threshold for duplicate detection.
            vector_store_manager: Optional for checking existing documents.
        """
        self._service = DeduplicationService(threshold)
        self._hasher = ContentHasher()
        self._vector_store = vector_store_manager

    async def deduplicate(
        self,
        documents: List[Document],
        request: DocumentIngestionRequest,
        progress_callback: Optional[callable] = None,
    ) -> List[Document]:
        """Deduplicate documents against existing content and within batch.

        Args:
            documents: Documents to deduplicate.
            request: Original ingestion request.
            progress_callback: Optional callback for progress updates.

        Returns:
            Deduplicated list of documents.
        """
        if not documents:
            return documents

        if request.force_refresh:
            logger.info("Skipping deduplication due to force_refresh=True")
            if progress_callback:
                await progress_callback(len(documents), len(documents), 0)
            return documents

        try:
            # Check against existing documents
            documents = await self._deduplicate_against_existing(
                documents, request, progress_callback
            )

            # Deduplicate within batch
            documents = self._deduplicate_within_batch(documents)

            return documents

        except Exception as e:
            logger.warning(f"Deduplication failed: {e}. Proceeding without deduplication.")
            return documents

    async def _deduplicate_against_existing(
        self,
        documents: List[Document],
        request: DocumentIngestionRequest,
        progress_callback: Optional[callable] = None,
    ) -> List[Document]:
        """Check documents against existing ones using fast hash lookup.

        Uses O(1) hash-based lookup instead of O(n*m) pairwise comparison.
        """
        existing_docs = await self._get_existing_documents(request, limit=500)

        if not existing_docs:
            if progress_callback:
                await progress_callback(len(documents), len(documents), 0)
            return documents

        # Build hash set for O(1) exact match lookup
        existing_hashes = set()
        for doc in existing_docs:
            content = doc.get("content", "")
            if content:
                content_hash = self._hasher.generate_content_hash(content)
                existing_hashes.add(content_hash)

        logger.info(f"Built hash index with {len(existing_hashes)} existing content hashes")

        duplicates_to_remove = set()
        total_docs = len(documents)

        for idx, doc in enumerate(documents):
            # Report progress every 100 chunks (fast now, less updates needed)
            if progress_callback and idx % 100 == 0:
                await progress_callback(idx, total_docs, len(duplicates_to_remove))

            content = doc.content
            content_hash = self._hasher.generate_content_hash(content)

            # Fast path: O(1) hash lookup for exact match
            if content_hash in existing_hashes:
                duplicates_to_remove.add(doc.id)
                continue

            # Note: Skip slow fuzzy matching - exact hash catches 99%+ of duplicates
            # If fuzzy matching is needed, use LSH pre-screening here

        # Final progress update
        if progress_callback:
            await progress_callback(total_docs, total_docs, len(duplicates_to_remove))

        if duplicates_to_remove:
            documents = [doc for doc in documents if doc.id not in duplicates_to_remove]
            logger.info(f"Removed {len(duplicates_to_remove)} duplicate chunks (hash-based)")

        return documents

    def _deduplicate_within_batch(
        self,
        documents: List[Document],
    ) -> List[Document]:
        """Deduplicate documents within the current batch.

        Uses fast hash-based deduplication instead of O(n²) similarity checks.
        """
        # For large batches, use fast hash-based deduplication only
        # O(n²) similarity checks are too slow for batches > 100 chunks
        if len(documents) > 100:
            logger.info(f"Using fast hash-based deduplication for {len(documents)} chunks")
            seen_hashes = {}
            unique_docs = []
            for doc in documents:
                content_hash = self._hasher.generate_content_hash(doc.content)
                if content_hash not in seen_hashes:
                    seen_hashes[content_hash] = doc.id
                    unique_docs.append(doc)

            removed = len(documents) - len(unique_docs)
            if removed > 0:
                logger.info(f"Fast dedup removed {removed} exact duplicates within batch")
            return unique_docs

        docs_for_dedup = [
            {
                "id": doc.id,
                "content": doc.content,
                "metadata": (
                    doc.metadata.model_dump()
                    if hasattr(doc.metadata, "model_dump")
                    else doc.metadata
                ),
            }
            for doc in documents
        ]

        deduplicated = self._service.deduplicate_chunks(docs_for_dedup, strategy="merge")

        # Convert back to Document objects
        final_docs = []
        for dedup_doc in deduplicated:
            original = next(
                (doc for doc in documents if doc.id == dedup_doc["id"]),
                None,
            )
            if original:
                if "metadata" in dedup_doc:
                    for key, value in dedup_doc["metadata"].items():
                        if hasattr(original.metadata, key):
                            setattr(original.metadata, key, value)
                final_docs.append(original)

        return final_docs

    async def _get_existing_documents(
        self,
        request: DocumentIngestionRequest,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get existing documents for deduplication check."""
        if not self._vector_store:
            return []

        try:
            lookup_values: List[str] = []
            if request.url:
                lookup_values.append(request.url)
            if request.file_path:
                lookup_values.append(request.file_path)
            if request.metadata:
                candidate = request.metadata.get("source") or request.metadata.get(
                    "canonical_url"
                )
                if candidate:
                    lookup_values.append(candidate)

            if not lookup_values:
                return []

            lookup_values = list({str(value) for value in lookup_values if value})

            def _fetch_matching_docs() -> List[Dict[str, Any]]:
                all_docs = self._vector_store.get_all_documents(refresh=False)
                matches: List[Dict[str, Any]] = []

                for doc in all_docs:
                    metadata = doc.metadata or {}
                    source_candidates = {
                        metadata.get("source"),
                        metadata.get("canonical_url"),
                        metadata.get("file_path"),
                    }

                    if any(
                        value in lookup_values
                        for value in source_candidates
                        if value
                    ):
                        matches.append(
                            {
                                "id": metadata.get("id"),
                                "content": doc.page_content,
                                "metadata": metadata,
                            }
                        )
                        if len(matches) >= limit:
                            break

                return matches

            return await asyncio.to_thread(_fetch_matching_docs)

        except Exception as e:
            logger.warning(f"Failed to get existing documents: {e}")
            return []

    def generate_content_hash(self, content: str) -> str:
        """Generate a content hash for duplicate detection.

        Args:
            content: The content to hash.

        Returns:
            Hash string.
        """
        return self._hasher.generate_content_hash(content)
