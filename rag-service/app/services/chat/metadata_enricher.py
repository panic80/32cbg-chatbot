"""Metadata enrichment service for kilometric rates and glossary injection."""

import os
from typing import Any, Dict, List, Optional, Tuple

from app.api.prompt_constants import GMT_GLOSSARY_NOTE
from app.core.logging import get_logger
from app.models.query import ChatRequest, Source
from app.utils.rate_tables import get_kilometric_rate, normalize_kilometric_location

logger = get_logger(__name__)

# Keywords that indicate kilometric/mileage rate queries
_KILOMETRIC_KEYWORDS = (
    "kilometric",
    "mileage",
    "mile rate",
    "cents/km",
    "cents per km",
    "cents per kilometre",
    "cents per kilometer",
    "per kilometre",
    "per kilometer",
)


def is_kilometric_query(*texts: Optional[str]) -> bool:
    """Determine whether any provided text is asking for a kilometric/mileage rate."""
    for text in texts:
        if not text:
            continue
        lowered = text.lower()
        if any(keyword in lowered for keyword in _KILOMETRIC_KEYWORDS):
            return True
    return False


def infer_kilometric_location(
    chat_request: ChatRequest,
    classification: Optional[Dict[str, Any]],
    optimized_query: str,
) -> Optional[str]:
    """Infer the jurisdiction associated with a kilometric rate question."""
    candidates: List[str] = []

    if classification:
        location_context = classification.get("location_context")
        if location_context:
            candidates.append(location_context)
        entities = classification.get("entities") or []
        candidates.extend(str(entity) for entity in entities if entity)

    candidates.append(chat_request.message)
    if chat_request.chat_history:
        candidates.extend(
            message.content
            for message in chat_request.chat_history
            if getattr(message, "role", None) == "user"
        )

    candidates.append(optimized_query)

    for candidate in candidates:
        normalized = normalize_kilometric_location(candidate)
        if normalized:
            return normalized
    return None


def clean_reference_label(value: Optional[str]) -> Optional[str]:
    """Normalize reference labels to avoid exposing internal file paths."""
    if not value:
        return None

    cleaned = value.strip()
    if not cleaned:
        return None

    if cleaned.startswith(("http://", "https://")):
        return cleaned

    # Strip known workspace prefixes
    workspace_prefixes = [
        "/var/www/cbthis/",
        os.getenv("RAG_WORKSPACE_ROOT", "").rstrip("/") + "/",
    ]
    for prefix in workspace_prefixes:
        if prefix and cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
            break

    cleaned = cleaned.lstrip("/\\")

    # Reduce to basename if path-like
    if "/" in cleaned or "\\" in cleaned:
        cleaned = os.path.basename(cleaned)

    return cleaned or None


def sanitize_source_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of metadata with any sensitive path values cleaned."""
    sanitized = dict(metadata or {})
    for key in ("source", "filename", "file_path", "path", "document_path"):
        value = sanitized.get(key)
        if isinstance(value, str):
            sanitized_value = clean_reference_label(value)
            if sanitized_value is None:
                sanitized.pop(key, None)
            else:
                sanitized[key] = sanitized_value
    return sanitized


class MetadataEnricher:
    """Enriches context and sources with additional metadata."""

    def enrich_with_kilometric_rate(
        self,
        context: str,
        sources: List[Source],
        chat_request: ChatRequest,
        optimized_query: str,
        classification: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, List[Source], bool]:
        """Add kilometric rate information if applicable.

        Args:
            context: The current context string.
            sources: List of sources.
            chat_request: The chat request.
            optimized_query: The optimized query.
            classification: Optional classification dict.

        Returns:
            Tuple of (enriched_context, enriched_sources, was_added)
        """
        if not is_kilometric_query(optimized_query, chat_request.message):
            return context, sources, False

        location_hint = infer_kilometric_location(
            chat_request, classification, optimized_query
        )
        query_for_lookup = location_hint or chat_request.message or optimized_query
        rate_info = get_kilometric_rate(query_for_lookup)

        if not rate_info:
            return context, sources, False

        raw_location, rate_value, effective_date = rate_info
        snippet = (
            f"{raw_location} PMV kilometric rate (taxes included) is "
            f"{rate_value:.1f} cents per kilometre (effective {effective_date})."
        )

        # Check if rate already in context
        if f"{rate_value:.1f}" in context:
            return context, sources, False

        logger.info("Injecting kilometric fallback snippet for %s", raw_location)
        metadata = sanitize_source_metadata({
            "source": "NJC Travel Directive Appendix B – Kilometric Rates",
            "content_type": "table_key_value",
            "location": raw_location,
            "rate_cents_per_km": rate_value,
            "effective_date": effective_date,
        })

        kilometric_source = Source(
            id=f"kilometric_{raw_location.lower().replace(' ', '_')}",
            source_id="kilometric_rates_appendix_b",
            text=snippet,
            title="NJC Travel Directive Appendix B – Kilometric Rates",
            url="https://www.njc-cnm.gc.ca/directive/d10/v238/en?print",
            section="Kilometric rates (Appendix B)",
            page=None,
            score=1.0,
            metadata=metadata,
        )

        enriched_sources = [kilometric_source] + (sources or [])
        enriched_context = f"{snippet}\n\n{context}" if context else snippet

        return enriched_context, enriched_sources, True

    def enrich_with_glossary(
        self,
        context: str,
        sources: List[Source],
        message: str,
    ) -> Tuple[str, List[Source], Optional[Source], bool]:
        """Add GMT glossary if applicable.

        Args:
            context: The current context string.
            sources: List of sources.
            message: The user's message.

        Returns:
            Tuple of (enriched_context, enriched_sources, glossary_source, was_injected)
        """
        if "gmt" not in (message or "").lower():
            return context, sources, None, False

        if "government motor transport" in context.lower():
            return context, sources, None, False

        glossary_block = f"[Glossary - Government Motor Transport]\n{GMT_GLOSSARY_NOTE}\n"
        enriched_context = f"{glossary_block}\n{context}" if context else glossary_block

        glossary_source = Source(
            id="glossary_gmt",
            source_id="glossary_gmt",
            text=GMT_GLOSSARY_NOTE,
            title="Government Motor Transport (GMT) definition",
            url=None,
            section="Glossary",
            page=None,
            score=1.0,
            metadata={
                "source": "cbthis glossary",
                "source_type": "glossary",
                "content_type": "definition",
                "tags": ["glossary", "gmt", "crown vehicle"],
            },
        )

        enriched_sources = [glossary_source] + (sources or [])
        return enriched_context, enriched_sources, glossary_source, True

    def append_glossary_to_response(
        self,
        response: str,
        glossary_injected: bool,
    ) -> Tuple[str, Optional[str]]:
        """Append glossary note to response if needed.

        Args:
            response: The LLM response.
            glossary_injected: Whether glossary was injected in context.

        Returns:
            Tuple of (final_response, appended_text_or_none)
        """
        if not glossary_injected:
            return response, None

        normalized_response = response.lower()
        if (
            "government motor transport" in normalized_response
            or "crown vehicle" in normalized_response
        ):
            return response, None

        glossary_note_text = f"\n\n**Glossary Note:** {GMT_GLOSSARY_NOTE}"
        return response + glossary_note_text, glossary_note_text
