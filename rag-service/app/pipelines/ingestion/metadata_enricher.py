"""Metadata extraction and enrichment for ingested documents."""

import hashlib
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document as LangchainDocument

from app.core.config import settings
from app.core.logging import get_logger
from app.models.documents import (
    Document,
    DocumentType,
    DocumentMetadata,
    DocumentIngestionRequest,
)
from app.services.metadata_extractor import MetadataExtractor

logger = get_logger(__name__)


class IngestionMetadataEnricher:
    """Enriches document metadata during ingestion."""

    def __init__(self, llm: Optional[Any] = None):
        """Initialize metadata enricher.

        Args:
            llm: Optional LLM for advanced metadata extraction.
        """
        self._extractor = MetadataExtractor(llm=llm)
        self._enable_extraction = getattr(settings, "enable_metadata_extraction", True)

    async def enrich_chunks(
        self,
        chunks: List[LangchainDocument],
        request: Optional[DocumentIngestionRequest] = None,
    ) -> List[LangchainDocument]:
        """Enrich chunk metadata.

        Args:
            chunks: Document chunks to enrich.
            request: Original ingestion request for context.

        Returns:
            Chunks with enriched metadata.
        """
        if not self._enable_extraction:
            return chunks

        enriched = []
        for chunk in chunks:
            try:
                extracted = await self._extractor.extract_metadata(chunk)
                chunk.metadata.update(extracted)
            except Exception as e:
                logger.warning(f"Failed to extract metadata for chunk: {e}")

            # Extract column numbers from DOA documents
            column_meta = self._extract_column_number(chunk)
            if column_meta:
                chunk.metadata.update(column_meta)

            enriched.append(chunk)

        # Add column context to table chunks
        if request:
            enriched = self._add_column_context(enriched, request)

        return enriched

    def _extract_column_number(
        self,
        chunk: LangchainDocument,
    ) -> Optional[Dict[str, Any]]:
        """Extract column number metadata from DOA document chunks."""
        text = chunk.page_content
        if not text:
            return None

        # Pattern for "Column X –" or "Column X:"
        column_pattern = r"Column\s+(\d+)\s+[–:-]\s+([^\n]+)"
        match = re.search(column_pattern, text)

        if match:
            column_num = match.group(1)
            column_name = match.group(2).strip()
            column_name = re.sub(r"[:\s]+$", "", column_name)

            return {
                "column_number": column_num,
                "column_name": column_name,
            }

        return None

    def _add_column_context(
        self,
        chunks: List[LangchainDocument],
        request: DocumentIngestionRequest,
    ) -> List[LangchainDocument]:
        """Add column headers to DOA table chunks."""
        source = request.metadata.get("source", "") if request.metadata else ""
        if "delegation" not in source.lower():
            return chunks

        column_patterns = {
            17: {
                "keywords": [
                    "Services (Competitive)",
                    "competitive) – general",
                    "competitive general",
                ],
                "anti_keywords": ["non-competitive"],
                "header": "Column 17 – Services (Competitive) – General:\n\n",
            },
            18: {
                "keywords": [
                    "Services (Non-Competitive)",
                    "non-competitive) – general",
                    "non-competitive general",
                ],
                "anti_keywords": [],
                "header": "Column 18 – Services (Non-Competitive) – General:\n\n",
            },
        }

        modified_count = 0
        for chunk in chunks:
            text = chunk.page_content
            text_lower = text.lower()

            if re.match(r"^Column\s+\d+\s+[–:-]", text):
                continue

            for col_num, pattern_info in column_patterns.items():
                has_keyword = any(
                    kw.lower() in text_lower for kw in pattern_info["keywords"]
                )
                has_anti = any(
                    akw.lower() in text_lower for akw in pattern_info["anti_keywords"]
                )

                if has_keyword and not has_anti:
                    chunk.page_content = pattern_info["header"] + text
                    chunk.metadata["column_number"] = str(col_num)
                    chunk.metadata["column_context_added"] = True
                    modified_count += 1
                    break

        if modified_count > 0:
            logger.info(f"Added column context to {modified_count} chunks")

        return chunks

    def convert_to_internal_documents(
        self,
        chunks: List[LangchainDocument],
        doc_id: str,
        request: DocumentIngestionRequest,
    ) -> List[Document]:
        """Convert LangChain documents to internal Document format.

        Args:
            chunks: LangChain document chunks.
            doc_id: Parent document ID.
            request: Original ingestion request.

        Returns:
            List of internal Document objects.
        """
        internal_docs = []

        for i, chunk in enumerate(chunks):
            try:
                source_fields = self._prepare_source_metadata(chunk.metadata, request)

                # Extract CLI-provided metadata
                source_identity = chunk.metadata.get("source_identity")
                document_info = chunk.metadata.get("document_info")
                structure_info = chunk.metadata.get("structure_info")

                metadata = DocumentMetadata(
                    source=source_fields.get("source", "direct_input"),
                    title=(
                        document_info.get("title")
                        if document_info and document_info.get("title")
                        else chunk.metadata.get("title")
                    ),
                    type=DocumentType(chunk.metadata.get("type", request.type)),
                    section=chunk.metadata.get("section"),
                    page_number=chunk.metadata.get("page"),
                    last_modified=chunk.metadata.get("last_modified"),
                    policy_reference=chunk.metadata.get("policy_reference"),
                    tags=chunk.metadata.get("tags", []),
                    source_id=source_fields.get("source_id"),
                    canonical_url=(
                        document_info.get("canonical_url")
                        if document_info and document_info.get("canonical_url")
                        else source_fields.get("canonical_url")
                    ),
                    reference_path=source_fields.get("reference_path"),
                    source_identity=source_identity,
                    document_info=document_info,
                    publisher_confidence=(
                        source_identity.get("confidence") if source_identity else None
                    ),
                    evidence=source_identity.get("evidence") if source_identity else None,
                    structure_info=structure_info,
                )

                internal_doc = Document(
                    id=f"{doc_id}_chunk_{i}",
                    content=chunk.page_content,
                    metadata=metadata,
                    chunk_index=i,
                    parent_id=doc_id,
                    created_at=datetime.now(timezone.utc),
                )
                internal_docs.append(internal_doc)

            except Exception as e:
                logger.warning(f"Failed to convert chunk {i}: {e}")

        return internal_docs

    def _prepare_source_metadata(
        self,
        raw_metadata: Dict[str, Any],
        request: Optional[DocumentIngestionRequest] = None,
    ) -> Dict[str, Optional[str]]:
        """Derive stable source metadata for cataloging."""
        metadata = raw_metadata or {}

        source_value = metadata.get("source")
        if not source_value and request:
            source_value = request.url or request.file_path or "direct_input"
        if not source_value:
            source_value = "direct_input"

        canonical_url = metadata.get("canonical_url")
        if (
            not canonical_url
            and isinstance(source_value, str)
            and source_value.startswith("http")
        ):
            canonical_url = source_value

        reference_path = (
            metadata.get("reference_path")
            or metadata.get("section")
            or metadata.get("title")
        )

        source_identifier = metadata.get("source_id")
        if not source_identifier:
            identifier_parts = [
                str(part)
                for part in [
                    canonical_url or source_value,
                    reference_path,
                    metadata.get("policy_reference"),
                    metadata.get("document_id"),
                    (request.type if request else metadata.get("type")),
                ]
                if part
            ]
            identifier_seed = "|".join(identifier_parts).strip()

            if not identifier_seed:
                inferred_type = (
                    request.type if request else metadata.get("type") or "unknown"
                )
                identifier_seed = f"{inferred_type}:{source_value}"

            source_identifier = hashlib.sha1(identifier_seed.encode("utf-8")).hexdigest()

        return {
            "source": str(source_value) if source_value else "direct_input",
            "canonical_url": str(canonical_url) if canonical_url else None,
            "reference_path": str(reference_path) if reference_path else None,
            "source_id": source_identifier,
        }
