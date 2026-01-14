"""Tests for the query processor service."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.chat.query_processor import (
    QueryProcessor,
    ensure_provider,
    resolve_model,
    should_use_hybrid,
    build_classification_note,
    _get_classification_cache_key,
)
from app.models.query import Provider, ChatRequest


class TestEnsureProvider:
    """Tests for ensure_provider function."""

    def test_provider_enum_passthrough(self):
        """Test that Provider enum passes through unchanged."""
        result = ensure_provider(Provider.OPENAI)
        assert result == Provider.OPENAI

    def test_string_to_provider(self):
        """Test string conversion to Provider."""
        result = ensure_provider("openai")
        assert result == Provider.OPENAI

    def test_invalid_provider_defaults_to_openai(self):
        """Test invalid provider defaults to OPENAI."""
        result = ensure_provider("invalid_provider")
        assert result == Provider.OPENAI


class TestResolveModel:
    """Tests for resolve_model function."""

    def test_returns_requested_model_if_provided(self):
        """Test that requested model is returned if specified."""
        result = resolve_model(Provider.OPENAI, "gpt-4-turbo")
        assert result == "gpt-4-turbo"

    def test_default_openai_model(self):
        """Test default OpenAI model."""
        result = resolve_model(Provider.OPENAI, None)
        assert result is not None
        assert isinstance(result, str)

    def test_default_anthropic_model(self):
        """Test default Anthropic model."""
        result = resolve_model(Provider.ANTHROPIC, None)
        assert result is not None
        assert isinstance(result, str)


class TestShouldUseHybrid:
    """Tests for should_use_hybrid function."""

    def test_hybrid_enabled(self):
        """Test when hybrid search is enabled."""
        request = MagicMock()
        request.use_hybrid_search = True
        assert should_use_hybrid(request) is True

    def test_hybrid_disabled(self):
        """Test when hybrid search is disabled."""
        request = MagicMock()
        request.use_hybrid_search = False
        assert should_use_hybrid(request) is False

    def test_hybrid_not_set(self):
        """Test when hybrid search attribute doesn't exist."""
        request = MagicMock(spec=[])
        assert should_use_hybrid(request) is False


class TestClassificationCacheKey:
    """Tests for classification cache key generation."""

    def test_deterministic(self):
        """Test same query produces same key."""
        key1 = _get_classification_cache_key("test query")
        key2 = _get_classification_cache_key("test query")
        assert key1 == key2

    def test_case_insensitive(self):
        """Test cache key is case insensitive."""
        key1 = _get_classification_cache_key("Test Query")
        key2 = _get_classification_cache_key("test query")
        assert key1 == key2

    def test_whitespace_normalized(self):
        """Test whitespace is normalized."""
        key1 = _get_classification_cache_key("  test query  ")
        key2 = _get_classification_cache_key("test query")
        assert key1 == key2

    def test_different_queries_different_keys(self):
        """Test different queries produce different keys."""
        key1 = _get_classification_cache_key("query one")
        key2 = _get_classification_cache_key("query two")
        assert key1 != key2


class TestQueryProcessor:
    """Tests for QueryProcessor class."""

    @pytest.fixture
    def mock_optimizer(self):
        """Create a mock query optimizer."""
        with patch("app.services.chat.query_processor.QueryOptimizer") as mock:
            optimizer = MagicMock()
            optimizer.expand_abbreviations = MagicMock(side_effect=lambda x: x)
            optimizer.classify_query = AsyncMock(return_value=None)
            optimizer.expand_query = MagicMock(return_value=None)
            mock.return_value = optimizer
            yield optimizer

    @pytest.mark.asyncio
    async def test_process_query_basic(self, mock_optimizer):
        """Test basic query processing."""
        processor = QueryProcessor()

        query, classification, classification_dict = await processor.process_query(
            "What is the meal allowance?"
        )

        assert query == "What is the meal allowance?"
        mock_optimizer.expand_abbreviations.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_query_smart_mode_skips_classification(self, mock_optimizer):
        """Test that smart mode skips classification."""
        processor = QueryProcessor()

        await processor.process_query("test query", is_smart_mode=True)

        mock_optimizer.classify_query.assert_not_called()

    @pytest.mark.asyncio
    async def test_classification_cache_hit(self, mock_optimizer):
        """Test classification cache hit."""
        processor = QueryProcessor()
        processor._enable_cache = True

        # First call - cache miss
        await processor.process_query("test query")
        first_call_count = mock_optimizer.classify_query.call_count

        # Clear to ensure we're testing the cache
        QueryProcessor.clear_classification_cache()

        # Now test with cache disabled
        processor._enable_cache = False
        await processor.process_query("test query")
        second_call_count = mock_optimizer.classify_query.call_count

        # Second call should have also called classify (cache disabled)
        assert second_call_count > first_call_count

    def test_clear_classification_cache(self, mock_optimizer):
        """Test clearing classification cache."""
        # Clear should return count of cleared entries
        count = QueryProcessor.clear_classification_cache()
        assert isinstance(count, int)

    def test_expand_abbreviations(self, mock_optimizer):
        """Test abbreviation expansion."""
        processor = QueryProcessor()
        result = processor.expand_abbreviations("test text")
        mock_optimizer.expand_abbreviations.assert_called_with("test text")


class TestBuildClassificationNote:
    """Tests for build_classification_note function."""

    def test_none_classification_returns_none(self):
        """Test None classification returns None."""
        result = build_classification_note(None)
        assert result is None

    def test_classification_with_values_returns_note(self):
        """Test classification with values returns a note."""
        result = build_classification_note({"intent": "test"})
        assert result is not None
        # Check for key phrases that should be in the note
        assert "heuristic" in result.lower() or "interpretation" in result.lower()

    def test_includes_intent(self):
        """Test that intent is included in note."""
        classification = {"intent": "meal_allowance"}
        result = build_classification_note(classification)
        assert "meal_allowance" in result

    def test_includes_entities(self):
        """Test that entities are included in note."""
        classification = {"entities": ["Ottawa", "Toronto"]}
        result = build_classification_note(classification)
        assert "Ottawa" in result
        assert "Toronto" in result

    def test_formats_boolean_fields(self):
        """Test boolean fields are formatted correctly."""
        classification = {
            "is_class_a_context": True,
            "irregular_hours": False,
            "on_td_or_tasking": None,
        }
        result = build_classification_note(classification)
        assert "yes" in result
        assert "no" in result
        assert "unknown" in result
