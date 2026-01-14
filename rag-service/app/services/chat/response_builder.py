"""Response building service for context and source formatting."""

import asyncio
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document

from app.components.result_processor import ResultProcessor
from app.core.config import settings
from app.core.logging import get_logger
from app.models.query import Source
from app.services.chat.metadata_enricher import (
    clean_reference_label,
    sanitize_source_metadata,
)

logger = get_logger(__name__)


class ResponseBuilder:
    """Builds context and formats sources for LLM response generation."""

    def __init__(self, result_processor: Optional[ResultProcessor] = None):
        """Initialize response builder.

        Args:
            result_processor: Optional result processor. Created if not provided.
        """
        self.result_processor = result_processor or ResultProcessor()

    async def build_context_and_sources(
        self,
        results: List[Tuple],
        query: str,
        vector_store_manager: Optional[Any] = None,
    ) -> Tuple[str, List[Source]]:
        """Build context string and source list from retrieval results.

        Args:
            results: List of (document, score) tuples.
            query: The search query.
            vector_store_manager: Optional for supplemental searches.

        Returns:
            Tuple of (context_string, sources_list)
        """
        if not results:
            return "", []

        documents = [doc for doc, _ in results]
        processed_docs = await asyncio.to_thread(
            self.result_processor.process_results, documents, query
        )

        # Boost column-specific chunks
        processed_docs = await self._boost_column_matches(
            documents, processed_docs, query, vector_store_manager
        )

        context_parts: List[str] = []
        sources: List[Source] = []
        max_length = getattr(settings, "source_preview_max_length", 700)

        for index, doc in enumerate(processed_docs):
            raw_metadata = getattr(doc, "metadata", {}) or {}
            metadata = sanitize_source_metadata(raw_metadata)
            score = metadata.get("score", 0.0)
            is_table_content = (
                "|" in doc.page_content
                or "table" in (metadata.get("content_type", "") or "").lower()
            )

            # Build context without source number labels
            if is_table_content:
                context_parts.append(f"[Table Content]\n{doc.page_content}\n")
            else:
                context_parts.append(f"{doc.page_content}\n")

            preview_text = doc.page_content
            if max_length > 0 and not is_table_content and len(preview_text) > max_length:
                preview_text = preview_text[:max_length] + "..."

            title_candidate = (
                metadata.get("title")
                or metadata.get("filename")
                or metadata.get("source")
            )
            sanitized_title = (
                clean_reference_label(title_candidate) or metadata.get("title")
            )

            url_candidate = metadata.get("canonical_url") or metadata.get("url")
            sanitized_url = (
                url_candidate
                if url_candidate and url_candidate.startswith(("http://", "https://"))
                else None
            )

            source = Source(
                id=metadata.get("id", f"source_{index}"),
                text=preview_text,
                title=sanitized_title,
                url=sanitized_url,
                section=metadata.get("section"),
                page=metadata.get("page_number"),
                score=score,
                metadata=metadata,
            )
            sources.append(source)

        return "\n".join(context_parts), sources

    async def _boost_column_matches(
        self,
        documents: List[Document],
        processed_docs: List[Document],
        query: str,
        vector_store_manager: Optional[Any],
    ) -> List[Document]:
        """Boost column-specific chunks in results.

        Args:
            documents: Original documents.
            processed_docs: Processed documents.
            query: The search query.
            vector_store_manager: For supplemental searches.

        Returns:
            Boosted list of processed documents.
        """
        column_matches = re.findall(r"column\s+(\d+)", query, flags=re.IGNORECASE)
        if not column_matches:
            return processed_docs

        column_terms = {f"column {match}".lower() for match in column_matches}

        def _contains_column_reference(doc: Document) -> bool:
            content_lower = doc.page_content.lower()
            return any(term in content_lower for term in column_terms)

        priority_candidates = [doc for doc in documents if _contains_column_reference(doc)]

        if not priority_candidates and vector_store_manager:
            supplemental: List[Document] = []
            for term in column_terms:
                try:
                    search_results = await vector_store_manager.search(term, k=5)
                except Exception as exc:
                    logger.debug(
                        "Column heuristic search failed for '%s': %s", term, exc
                    )
                    continue
                for doc, _score in search_results:
                    if _contains_column_reference(doc):
                        supplemental.append(doc)
            if supplemental:
                priority_candidates = supplemental

        if priority_candidates:
            priority_processed = await asyncio.to_thread(
                self.result_processor.process_results,
                priority_candidates,
                query,
            )
            existing_ids = {doc.metadata.get("id") for doc in processed_docs}
            prioritized = []
            for doc in priority_processed:
                doc_id = doc.metadata.get("id")
                if doc_id in existing_ids:
                    continue
                existing_ids.add(doc_id)
                prioritized.append(doc)
            if prioritized:
                return prioritized + processed_docs

        return processed_docs

    def build_citation_guide(self, sources: List[Source]) -> str:
        """Build citation guide from sources.

        Args:
            sources: List of sources.

        Returns:
            Citation guide string.
        """
        citation_entries = []

        for source in sources:
            if not source.title:
                continue

            citation_parts = [source.title]

            # Use structure_info if available
            structure_info = (
                source.metadata.get("structure_info") if hasattr(source, "metadata") else None
            )

            if structure_info and structure_info.get("has_structure"):
                # Add chapter if detected
                chapters = structure_info.get("chapters", [])
                if chapters:
                    chapter_num = chapters[0].get("number")
                    if chapter_num:
                        citation_parts.append(f"Chapter {chapter_num}")

                # Add section if detected
                sections = structure_info.get("sections", [])
                if sections:
                    section_num = sections[0].get("number")
                    section_title = sections[0].get("title")
                    if section_num:
                        if section_title:
                            citation_parts.append(
                                f"Section {section_num} ({section_title})"
                            )
                        else:
                            citation_parts.append(f"Section {section_num}")

                # Add paragraph number
                paragraphs = structure_info.get("paragraphs", [])
                if paragraphs:
                    para_num = paragraphs[0].get("number")
                    if para_num:
                        citation_parts.append(f"paragraph {para_num}")

            elif source.section:
                citation_parts.append(f"section {source.section}")

            if source.page:
                citation_parts.append(f"page {source.page}")

            citation_entries.append("- " + ", ".join(citation_parts))

        if citation_entries:
            return (
                "\n\nCITATION GUIDE (use these for references):\n"
                + "\n".join(citation_entries)
                + "\n"
            )
        return ""

    def truncate_context(self, context: str, char_limit: int) -> str:
        """Truncate context to character limit.

        Args:
            context: The context string.
            char_limit: Maximum characters.

        Returns:
            Truncated context.
        """
        if char_limit and len(context) > char_limit:
            return context[:char_limit]
        return context
