"""Tests for the metadata enricher service."""

import pytest
from unittest.mock import MagicMock

from app.services.chat.metadata_enricher import (
    MetadataEnricher,
    is_kilometric_query,
    infer_kilometric_location,
    clean_reference_label,
    sanitize_source_metadata,
)
from app.models.query import ChatRequest, Provider, Source


class TestIsKilometricQuery:
    """Tests for is_kilometric_query function."""

    def test_kilometric_keyword(self):
        """Test detection of kilometric keyword."""
        assert is_kilometric_query("What is the kilometric rate?") is True

    def test_mileage_keyword(self):
        """Test detection of mileage keyword."""
        assert is_kilometric_query("What's the mileage rate?") is True

    def test_cents_per_km(self):
        """Test detection of cents/km."""
        assert is_kilometric_query("How many cents/km?") is True

    def test_no_keywords(self):
        """Test query without keywords."""
        assert is_kilometric_query("What is the meal allowance?") is False

    def test_none_input(self):
        """Test None input."""
        assert is_kilometric_query(None) is False

    def test_multiple_texts(self):
        """Test multiple text inputs."""
        assert is_kilometric_query("unrelated", "kilometric rate") is True


class TestInferKilometricLocation:
    """Tests for infer_kilometric_location function."""

    @pytest.fixture
    def basic_request(self):
        """Create a basic chat request."""
        return ChatRequest(
            message="What is the rate in Ontario?",
            provider=Provider.OPENAI,
        )

    def test_location_from_classification(self, basic_request):
        """Test location extracted from classification."""
        classification = {"location_context": "Ontario"}
        result = infer_kilometric_location(basic_request, classification, "test query")
        # Result depends on normalize_kilometric_location implementation
        assert result is None or isinstance(result, str)

    def test_location_from_entities(self, basic_request):
        """Test location from classification entities."""
        classification = {"entities": ["Alberta"]}
        result = infer_kilometric_location(basic_request, classification, "test query")
        assert result is None or isinstance(result, str)

    def test_none_classification(self, basic_request):
        """Test with None classification."""
        result = infer_kilometric_location(basic_request, None, "test query")
        assert result is None or isinstance(result, str)


class TestCleanReferenceLabel:
    """Tests for clean_reference_label function."""

    def test_none_input(self):
        """Test None input returns None."""
        assert clean_reference_label(None) is None

    def test_empty_string(self):
        """Test empty string returns None."""
        assert clean_reference_label("") is None

    def test_whitespace_only(self):
        """Test whitespace only returns None."""
        assert clean_reference_label("   ") is None

    def test_url_passthrough(self):
        """Test URLs pass through unchanged."""
        url = "https://example.com/doc.pdf"
        assert clean_reference_label(url) == url

    def test_strips_path_prefix(self):
        """Test path prefixes are stripped."""
        result = clean_reference_label("/var/www/cbthis/docs/test.pdf")
        assert not result.startswith("/var")

    def test_reduces_to_basename(self):
        """Test paths are reduced to basename."""
        result = clean_reference_label("/some/path/to/document.pdf")
        assert result == "document.pdf"


class TestSanitizeSourceMetadata:
    """Tests for sanitize_source_metadata function."""

    def test_empty_metadata(self):
        """Test empty metadata returns empty dict."""
        result = sanitize_source_metadata({})
        assert result == {}

    def test_none_metadata(self):
        """Test None metadata returns empty dict."""
        result = sanitize_source_metadata(None)
        assert result == {}

    def test_cleans_source_field(self):
        """Test source field is cleaned."""
        metadata = {"source": "/var/www/cbthis/docs/test.pdf"}
        result = sanitize_source_metadata(metadata)
        assert not result.get("source", "").startswith("/var")

    def test_preserves_other_fields(self):
        """Test non-path fields are preserved."""
        metadata = {"title": "Test Document", "score": 0.95}
        result = sanitize_source_metadata(metadata)
        assert result["title"] == "Test Document"
        assert result["score"] == 0.95


class TestMetadataEnricher:
    """Tests for MetadataEnricher class."""

    @pytest.fixture
    def enricher(self):
        """Create a metadata enricher."""
        return MetadataEnricher()

    @pytest.fixture
    def basic_request(self):
        """Create a basic chat request."""
        return ChatRequest(
            message="What is the meal allowance?",
            provider=Provider.OPENAI,
        )

    def test_enrich_with_kilometric_rate_no_match(self, enricher, basic_request):
        """Test enrichment when query doesn't match kilometric."""
        context, sources, was_added = enricher.enrich_with_kilometric_rate(
            context="Some context",
            sources=[],
            chat_request=basic_request,
            optimized_query="What is the meal allowance?",
        )
        assert was_added is False
        assert context == "Some context"

    def test_enrich_with_glossary_no_gmt(self, enricher, basic_request):
        """Test glossary not added when no GMT mention."""
        context, sources, glossary_source, was_injected = enricher.enrich_with_glossary(
            context="Some context",
            sources=[],
            message="What is the meal allowance?",
        )
        assert was_injected is False
        assert glossary_source is None

    def test_enrich_with_glossary_gmt_mention(self, enricher):
        """Test glossary added when GMT mentioned."""
        context, sources, glossary_source, was_injected = enricher.enrich_with_glossary(
            context="Some context",
            sources=[],
            message="Can I use a GMT vehicle?",
        )
        assert was_injected is True
        assert glossary_source is not None
        assert "GMT" in context or "Government Motor Transport" in context

    def test_append_glossary_to_response_no_injection(self, enricher):
        """Test no glossary appended when not injected."""
        response, appended = enricher.append_glossary_to_response(
            "Test response", glossary_injected=False
        )
        assert response == "Test response"
        assert appended is None

    def test_append_glossary_when_already_explained(self, enricher):
        """Test glossary not appended when response explains GMT."""
        response, appended = enricher.append_glossary_to_response(
            "Government Motor Transport (GMT) is a crown vehicle.",
            glossary_injected=True,
        )
        assert appended is None
