"""Query processing service for classification and optimization."""

import hashlib
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import settings
from app.core.logging import get_logger
from app.models.query import ChatRequest, Provider
from app.pipelines.query_optimizer import QueryOptimizer, QueryClassification

logger = get_logger(__name__)

# In-memory LRU cache for classification results (fast path)
# Caches up to 1000 recent classifications
_classification_cache: Dict[str, Tuple[QueryClassification, Dict[str, Any]]] = {}
_CLASSIFICATION_CACHE_SIZE = 1000


def _get_classification_cache_key(query: str) -> str:
    """Generate cache key for query classification."""
    normalized = query.strip().lower()
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def ensure_provider(provider: Provider | str) -> Provider:
    """Ensure provider is a valid Provider enum."""
    if isinstance(provider, Provider):
        return provider
    try:
        return Provider(provider)
    except ValueError:
        logger.warning("Unknown provider %s, defaulting to OPENAI", provider)
        return Provider.OPENAI


def resolve_model(provider: Provider, requested_model: Optional[str]) -> str:
    """Resolve the model name based on provider and request."""
    if requested_model:
        return requested_model
    if provider == Provider.OPENAI:
        return getattr(settings, "openai_chat_model", "gpt-4.1-mini")
    if provider == Provider.ANTHROPIC:
        return getattr(settings, "anthropic_chat_model", "claude-3-sonnet-20240229")
    if provider == Provider.GOOGLE:
        return getattr(settings, "google_chat_model", "gemini-pro")
    return "default"


def should_use_hybrid(chat_request: ChatRequest) -> bool:
    """Check if hybrid search should be used."""
    return getattr(chat_request, "use_hybrid_search", False)


class QueryProcessor:
    """Handles query optimization and classification with caching."""

    def __init__(
        self,
        llm: Optional[Any] = None,
        cache_service: Optional[Any] = None,
    ):
        """Initialize query processor.

        Args:
            llm: Optional LLM for query optimization. If None, some
                 optimization features are disabled.
            cache_service: Optional cache service for persistent classification cache.
        """
        self.optimizer = QueryOptimizer(llm)
        self._cache_service = cache_service
        self._enable_cache = getattr(settings, "cache_query_classification", True)

    async def process_query(
        self,
        message: str,
        is_fast_mode: bool = False
    ) -> tuple[str, Optional[QueryClassification], Optional[Dict[str, Any]]]:
        """Process and optimize a query.

        Args:
            message: The user's query message.
            is_fast_mode: Whether to skip classification for fast mode.

        Returns:
            Tuple of (optimized_query, classification, classification_dict)
        """
        optimized_query = message
        classification: Optional[QueryClassification] = None
        classification_dict: Optional[Dict[str, Any]] = None

        try:
            # Expand abbreviations
            optimized_query = self.optimizer.expand_abbreviations(message)

            # Classify query (skip for fast mode - fast mode uses optimized path)
            if not is_fast_mode:
                classification, classification_dict = await self._get_classification(
                    optimized_query
                )

                # Expand query based on intent
                if classification and classification.intent != "unknown":
                    expanded = self.optimizer.expand_query(
                        optimized_query, classification.intent
                    )
                    if expanded:
                        optimized_query = expanded[0]

            # Add default location if not present
            if (
                not getattr(classification, "location_context", None)
                and getattr(settings, "default_location", None)
            ):
                optimized_query = f"{optimized_query} {settings.default_location}"

        except Exception as exc:
            logger.warning("Query optimisation skipped due to error: %s", exc)

        return optimized_query, classification, classification_dict

    async def _get_classification(
        self,
        query: str,
    ) -> Tuple[Optional[QueryClassification], Optional[Dict[str, Any]]]:
        """Get classification with caching.

        Checks in-memory cache first, then Redis cache, then computes.
        """
        global _classification_cache

        if not self._enable_cache:
            return await self._compute_classification(query)

        cache_key = _get_classification_cache_key(query)

        # Check in-memory cache first (fastest)
        if cache_key in _classification_cache:
            logger.debug("Classification cache hit (memory): %s", cache_key[:8])
            return _classification_cache[cache_key]

        # Check Redis cache if available
        if self._cache_service:
            try:
                cached = await self._cache_service.get_classification(query)
                if cached:
                    logger.debug("Classification cache hit (Redis): %s", cache_key[:8])
                    # Reconstruct QueryClassification from dict
                    classification = QueryClassification(**cached)
                    result = (classification, cached)
                    # Update in-memory cache
                    self._update_memory_cache(cache_key, result)
                    return result
            except Exception as e:
                logger.debug("Redis classification cache error: %s", e)

        # Compute classification
        classification, classification_dict = await self._compute_classification(query)

        if classification and classification_dict:
            # Store in caches
            self._update_memory_cache(cache_key, (classification, classification_dict))

            if self._cache_service:
                try:
                    await self._cache_service.set_classification(
                        query, classification_dict
                    )
                except Exception as e:
                    logger.debug("Failed to cache classification: %s", e)

        return classification, classification_dict

    async def _compute_classification(
        self,
        query: str,
    ) -> Tuple[Optional[QueryClassification], Optional[Dict[str, Any]]]:
        """Compute query classification using the optimizer."""
        classification = await self.optimizer.classify_query(query)
        if classification:
            return classification, classification.model_dump()
        return None, None

    def _update_memory_cache(
        self,
        key: str,
        value: Tuple[QueryClassification, Dict[str, Any]],
    ) -> None:
        """Update in-memory cache with LRU eviction."""
        global _classification_cache

        # Simple LRU: if at capacity, remove oldest entry
        if len(_classification_cache) >= _CLASSIFICATION_CACHE_SIZE:
            # Remove first (oldest) item
            oldest_key = next(iter(_classification_cache))
            del _classification_cache[oldest_key]

        _classification_cache[key] = value

    def expand_abbreviations(self, text: str) -> str:
        """Expand known abbreviations in text."""
        return self.optimizer.expand_abbreviations(text)

    @classmethod
    def clear_classification_cache(cls) -> int:
        """Clear the in-memory classification cache.

        Returns:
            Number of entries cleared.
        """
        global _classification_cache
        count = len(_classification_cache)
        _classification_cache.clear()
        return count


def build_classification_note(classification: Optional[Dict[str, Any]]) -> Optional[str]:
    """Summarize query classification flags for downstream instructions."""
    if not classification:
        return None

    def _format_bool(value: Any) -> str:
        if value is None:
            return "unknown"
        return "yes" if bool(value) else "no"

    bool_fields = [
        ("Class A context", classification.get("is_class_a_context")),
        ("Working irregular hours", classification.get("irregular_hours")),
        ("Ordered outside normal hours", classification.get("ordered_outside_normal_hours")),
        ("On TD/tasking/MTEC", classification.get("on_td_or_tasking")),
        ("Missed meal on tasking", classification.get("missed_meal_on_tasking")),
        ("Entitlement likely denied", classification.get("entitlement_likely_denied")),
    ]

    lines = [
        "Use this heuristic interpretation of the user's scenario "
        "(derived from the question; confirm against official policy sources):"
    ]
    for label, value in bool_fields:
        lines.append(f"- {label}: {_format_bool(value)}")

    intent = classification.get("intent")
    if intent:
        lines.append(f"- Detected intent: {intent}")

    entities = classification.get("entities") or []
    if entities:
        cleaned_entities = sorted({str(entity) for entity in entities if entity})
        if cleaned_entities:
            lines.append(f"- Detected entities: {', '.join(cleaned_entities)}")

    return "\n".join(lines)
