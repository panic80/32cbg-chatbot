"""HyDE (Hypothetical Document Embeddings) Generator for improved retrieval."""

import asyncio
import hashlib
from typing import Optional

from app.components.base import BaseComponent
from app.core.config import settings
from app.core.logging import get_logger
from app.models.query import Provider

logger = get_logger(__name__)


HYDE_PROMPT_TEMPLATE = """You are an expert on Canadian Forces travel policies and instructions.
Generate a hypothetical but realistic answer to this question as if you were quoting directly from official CF travel documentation.
The answer should:
- Contain specific details, rates, or policy references that would be found in official documentation
- Be written in the formal style of government policy documents
- Include relevant section references, allowance amounts, or eligibility criteria if applicable
- Be concise (2-4 sentences)

Question: {query}

Hypothetical Policy Excerpt:"""


class HyDEGenerator(BaseComponent):
    """
    Generates hypothetical document embeddings for improved retrieval.

    HyDE works by generating a hypothetical answer to a query, then using
    that answer's embedding for retrieval. This bridges the semantic gap
    between questions and answers.
    """

    def __init__(self, llm_pool=None, cache_service=None):
        """
        Initialize the HyDE generator.

        Args:
            llm_pool: LLM connection pool for generating hypotheses
            cache_service: Optional LayeredCacheService for Redis caching
        """
        super().__init__(
            component_type="hyde",
            component_name="HyDEGenerator"
        )
        self.llm_pool = llm_pool
        self.cache_service = cache_service
        self._cache = {}  # In-memory cache for session

    def _get_cache_key(self, query: str) -> str:
        """Generate cache key for a query."""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()

    async def generate_hypothesis(
        self,
        query: str,
        use_cache: bool = True,
        model: Optional[str] = None,
        provider: Optional[Provider] = None
    ) -> Optional[str]:
        """
        Generate a hypothetical answer for a query.

        Args:
            query: The user's query
            use_cache: Whether to use cached results

        Returns:
            Hypothetical answer string, or None if generation fails
        """
        if not settings.enable_hyde:
            return None

        # Check in-memory cache first
        cache_key = self._get_cache_key(query)
        if use_cache and cache_key in self._cache:
            self._log_event("cache_hit", {"query": query[:50], "source": "memory"})
            return self._cache[cache_key]

        # Check Redis cache if available
        if use_cache and self.cache_service:
            try:
                cached = await self.cache_service.get_hyde_hypothesis(query)
                if cached:
                    self._cache[cache_key] = cached  # Also populate memory cache
                    self._log_event("cache_hit", {"query": query[:50], "source": "redis"})
                    return cached
            except Exception as e:
                logger.debug(f"Redis cache lookup failed: {e}")

        if not self.llm_pool:
            logger.warning("HyDE generator has no LLM pool configured")
            return None

        try:
            # Acquire LLM from pool
            async with self.llm_pool.acquire(
                provider or Provider.OPENAI,
                model or settings.hyde_model
            ) as llm:
                if not llm:
                    logger.error("Failed to acquire LLM for HyDE generation")
                    return None

                prompt = HYDE_PROMPT_TEMPLATE.format(query=query)

                # Generate with timeout
                try:
                    response = await asyncio.wait_for(
                        llm.ainvoke(prompt),
                        timeout=settings.hyde_timeout
                    )

                    hypothesis = response.content.strip()

                    # Cache the result in memory
                    if use_cache:
                        self._cache[cache_key] = hypothesis

                    # Cache in Redis if available
                    if use_cache and self.cache_service:
                        try:
                            await self.cache_service.set_hyde_hypothesis(query, hypothesis)
                        except Exception as e:
                            logger.debug(f"Redis cache store failed: {e}")

                    self._log_event(
                        "hypothesis_generated",
                        {
                            "query": query[:50],
                            "hypothesis_length": len(hypothesis)
                        }
                    )

                    return hypothesis

                except asyncio.TimeoutError:
                    self._log_event(
                        "timeout",
                        {"query": query[:50], "timeout": settings.hyde_timeout},
                        level="warning"
                    )
                    return None

        except Exception as e:
            self._log_event(
                "generation_error",
                {"query": query[:50], "error": str(e)},
                level="error"
            )
            return None

    async def generate_hypothesis_with_fallback(
        self,
        query: str,
        use_cache: bool = True
    ) -> str:
        """
        Generate hypothesis with fallback to original query.

        Args:
            query: The user's query
            use_cache: Whether to use cached results

        Returns:
            Hypothetical answer or original query if generation fails
        """
        hypothesis = await self.generate_hypothesis(query, use_cache)
        return hypothesis if hypothesis else query

    def clear_cache(self):
        """Clear the in-memory hypothesis cache."""
        self._cache.clear()
        self._log_event("cache_cleared", {"entries_cleared": len(self._cache)})


# Singleton instance
_hyde_generator: Optional[HyDEGenerator] = None


def get_hyde_generator(llm_pool=None, cache_service=None) -> HyDEGenerator:
    """
    Get or create the HyDE generator singleton.

    Args:
        llm_pool: LLM connection pool (required on first call)
        cache_service: Optional LayeredCacheService for Redis caching

    Returns:
        HyDEGenerator instance
    """
    global _hyde_generator

    if _hyde_generator is None:
        _hyde_generator = HyDEGenerator(llm_pool=llm_pool, cache_service=cache_service)
    else:
        if llm_pool is not None and _hyde_generator.llm_pool is None:
            _hyde_generator.llm_pool = llm_pool
        if cache_service is not None and _hyde_generator.cache_service is None:
            _hyde_generator.cache_service = cache_service

    return _hyde_generator
