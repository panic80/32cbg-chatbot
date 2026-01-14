"""Quality validation service for document chunks."""

import re
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from langchain_core.documents import Document
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

from app.core.logging import get_logger

logger = get_logger(__name__)

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class QualityLevel(str, Enum):
    """Quality levels for chunks."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INVALID = "invalid"


@dataclass
class QualityMetrics:
    """Quality metrics for a chunk."""
    quality_level: QualityLevel
    quality_score: float  # 0-100
    issues: List[str]
    metrics: Dict[str, Any]
    suggestions: List[str]


class ChunkQualityValidator:
    """Validate quality of document chunks."""
    
    def __init__(
        self,
        min_chunk_size: int = 50,
        max_chunk_size: int = 2000,
        min_quality_score: float = 60.0
    ):
        """Initialize quality validator.
        
        Args:
            min_chunk_size: Minimum acceptable chunk size
            max_chunk_size: Maximum acceptable chunk size
            min_quality_score: Minimum quality score to pass
        """
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.min_quality_score = min_quality_score
        
    def validate_chunk(self, chunk: Document) -> QualityMetrics:
        """Validate a single chunk and return quality metrics.
        
        Args:
            chunk: Document chunk to validate
            
        Returns:
            QualityMetrics object
        """
        content = chunk.page_content
        metadata = chunk.metadata
        
        # Calculate various quality metrics
        size_score = self._calculate_size_score(content)
        coherence_score = self._calculate_coherence_score(content)
        completeness_score = self._calculate_completeness_score(content)
        formatting_score = self._calculate_formatting_score(content)
        informativeness_score = self._calculate_informativeness_score(content)

        # Certain structured payloads (e.g., table serializations) should not be
        # penalized for low narrative coherence. Detect them early so we can
        # relax specific heuristics below.
        content_type = str(metadata.get("content_type", "")) if metadata else ""
        is_structured_table = content_type.startswith("table") or "table_title" in (metadata or {})

        if is_structured_table:
            # Structured tables often appear as JSON blobs which naturally score
            # low on coherence/informativeness. Bump the raw metrics so the
            # aggregated quality score reflects their utility.
            coherence_score = max(coherence_score, 75)
            informativeness_score = max(informativeness_score, 70)
        
        # Check for specific issues
        issues = []
        suggestions = []
        
        # Size issues
        if len(content) < self.min_chunk_size:
            issues.append(f"Chunk too small ({len(content)} chars)")
            suggestions.append("Merge with adjacent chunks")
        elif len(content) > self.max_chunk_size:
            issues.append(f"Chunk too large ({len(content)} chars)")
            suggestions.append("Split into smaller chunks")
            
        # Coherence issues
        if coherence_score < 50 and not is_structured_table:
            issues.append("Low coherence - may be truncated")
            suggestions.append("Check chunk boundaries")

        # Completeness issues
        if completeness_score < 50:
            issues.append("Incomplete content detected")
            suggestions.append("Extend chunk to include complete thoughts")

        # Formatting issues
        if formatting_score < 50 and not is_structured_table:
            issues.append("Poor formatting or structure")
            suggestions.append("Clean up formatting and structure")

        # Informativeness issues
        if informativeness_score < 30 and not is_structured_table:
            issues.append("Low information density")
            suggestions.append("Consider removing or merging chunk")
            
        # Calculate overall quality score
        quality_score = (
            size_score * 0.2 +
            coherence_score * 0.3 +
            completeness_score * 0.2 +
            formatting_score * 0.1 +
            informativeness_score * 0.2
        )
        
        # Determine quality level
        if is_structured_table:
            # Tables tend to be dense once normalized; treat them as at least
            # medium quality so they are available to downstream retrieval.
            quality_score = max(quality_score, self.min_quality_score + 5)

        if quality_score >= 80:
            quality_level = QualityLevel.HIGH
        elif quality_score >= 60:
            quality_level = QualityLevel.MEDIUM
        elif quality_score >= 40:
            quality_level = QualityLevel.LOW
        else:
            quality_level = QualityLevel.INVALID
            
        # Add metadata quality check
        metadata_score = self._validate_metadata(metadata)
        if metadata_score < 50:
            issues.append("Insufficient metadata")
            suggestions.append("Enrich metadata with source, type, and context")
            
        return QualityMetrics(
            quality_level=quality_level,
            quality_score=round(quality_score, 1),
            issues=issues,
            metrics={
                "size_score": round(size_score, 1),
                "coherence_score": round(coherence_score, 1),
                "completeness_score": round(completeness_score, 1),
                "formatting_score": round(formatting_score, 1),
                "informativeness_score": round(informativeness_score, 1),
                "metadata_score": round(metadata_score, 1),
                "char_count": len(content),
                "word_count": len(word_tokenize(content)),
                "sentence_count": len(sent_tokenize(content))
            },
            suggestions=suggestions
        )
        
    def validate_batch(self, chunks: List[Document]) -> Tuple[List[Document], List[Document], Dict[str, Any]]:
        """Validate a batch of chunks.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            Tuple of (valid_chunks, invalid_chunks, statistics)
        """
        valid_chunks = []
        invalid_chunks = []
        
        quality_scores = []
        quality_levels = {level: 0 for level in QualityLevel}
        all_issues = []
        
        for chunk in chunks:
            metrics = self.validate_chunk(chunk)
            
            # Add quality metrics to chunk metadata
            chunk.metadata["quality_score"] = metrics.quality_score
            chunk.metadata["quality_level"] = metrics.quality_level
            chunk.metadata["quality_issues"] = metrics.issues
            
            quality_scores.append(metrics.quality_score)
            quality_levels[metrics.quality_level] += 1
            all_issues.extend(metrics.issues)
            
            if metrics.quality_score >= self.min_quality_score:
                valid_chunks.append(chunk)
            else:
                invalid_chunks.append(chunk)
                
        # Calculate statistics
        statistics = {
            "total_chunks": len(chunks),
            "valid_chunks": len(valid_chunks),
            "invalid_chunks": len(invalid_chunks),
            "average_quality_score": round(sum(quality_scores) / len(quality_scores), 1) if quality_scores else 0,
            "quality_distribution": quality_levels,
            "common_issues": self._get_common_issues(all_issues),
            "validation_rate": round(len(valid_chunks) / len(chunks) * 100, 1) if chunks else 0
        }
        
        return valid_chunks, invalid_chunks, statistics
        
    def _calculate_size_score(self, content: str) -> float:
        """Calculate score based on chunk size."""
        char_count = len(content)
        
        if self.min_chunk_size <= char_count <= self.max_chunk_size:
            # Optimal size range
            optimal_size = (self.min_chunk_size + self.max_chunk_size) / 2
            deviation = abs(char_count - optimal_size) / optimal_size
            return max(0, 100 - deviation * 50)
        elif char_count < self.min_chunk_size:
            # Too small
            return max(0, (char_count / self.min_chunk_size) * 50)
        else:
            # Too large
            return max(0, 50 - ((char_count - self.max_chunk_size) / self.max_chunk_size) * 50)
            
    def _calculate_coherence_score(self, content: str) -> float:
        """Calculate coherence score based on sentence structure."""
        sentences = sent_tokenize(content)
        
        if not sentences:
            return 0
            
        # Check for incomplete sentences
        incomplete_count = 0
        for sentence in sentences:
            # Check if sentence ends properly
            if not re.search(r'[.!?]$', sentence.strip()):
                incomplete_count += 1
            # Check if sentence is too short
            if len(sentence.split()) < 3:
                incomplete_count += 0.5
                
        # Check for truncated start/end
        truncation_penalty = 0
        if not re.match(r'^[A-Z]', content.strip()):
            truncation_penalty += 10  # Doesn't start with capital
        if content.strip() and not re.search(r'[.!?]$', content.strip()):
            truncation_penalty += 10  # Doesn't end with punctuation
            
        coherence_score = max(0, 100 - (incomplete_count / len(sentences)) * 50 - truncation_penalty)
        return coherence_score
        
    def _calculate_completeness_score(self, content: str) -> float:
        """Calculate completeness score."""
        # Check for common truncation indicators
        truncation_patterns = [
            r'\.\.\.$',  # Ellipsis at end
            r'…$',        # Unicode ellipsis at end
            r'\[(continued|cont\.?)\]',
            r'\bmid-sentence\b',
            r'\b(etc|et cetera)$'
        ]
        
        truncation_count = sum(1 for pattern in truncation_patterns if re.search(pattern, content, re.IGNORECASE))
        
        # Check for balanced parentheses and quotes
        balance_score = 100
        if content.count('(') != content.count(')'):
            balance_score -= 20
        if content.count('"') % 2 != 0:
            balance_score -= 20
        if content.count("'") % 2 != 0:
            balance_score -= 10
            
        completeness_score = max(0, balance_score - truncation_count * 30)
        return completeness_score
        
    def _calculate_formatting_score(self, content: str) -> float:
        """Calculate formatting quality score."""
        formatting_score = 100
        
        # Check for excessive whitespace
        if re.search(r'\s{3,}', content):
            formatting_score -= 10
            
        # Check for malformed characters
        if re.search(r'[^\x00-\x7F]+', content):  # Non-ASCII might indicate encoding issues
            non_ascii_ratio = len(re.findall(r'[^\x00-\x7F]', content)) / len(content)
            if non_ascii_ratio > 0.1:  # More than 10% non-ASCII
                formatting_score -= 20
                
        # Check for repeated punctuation
        if re.search(r'[.!?]{3,}', content):
            formatting_score -= 10
            
        # Check for proper spacing around punctuation
        if re.search(r'\w[.!?]\w', content):  # No space after sentence end
            formatting_score -= 10
            
        # Check line length variation (indicates structure)
        lines = content.split('\n')
        if len(lines) > 1:
            line_lengths = [len(line) for line in lines if line.strip()]
            if line_lengths:
                avg_length = sum(line_lengths) / len(line_lengths)
                if all(abs(length - avg_length) < 10 for length in line_lengths):
                    formatting_score -= 10  # Too uniform, might be poorly formatted
                    
        return max(0, formatting_score)
        
    def _calculate_informativeness_score(self, content: str) -> float:
        """Calculate information density score."""
        words = word_tokenize(content.lower())
        
        if not words:
            return 0
            
        # Calculate unique word ratio
        unique_ratio = len(set(words)) / len(words)
        
        # Check for meaningful content (not just boilerplate)
        boilerplate_phrases = [
            "click here", "see below", "as follows", "page break",
            "continued from", "table of contents", "this page"
        ]
        boilerplate_count = sum(1 for phrase in boilerplate_phrases if phrase in content.lower())
        
        # Check for substantive words (longer words tend to be more informative)
        substantive_words = [w for w in words if len(w) > 4]
        substantive_ratio = len(substantive_words) / len(words)
        
        # Check for numbers and specific data
        has_numbers = bool(re.search(r'\d+', content))
        has_specific_data = bool(re.search(r'\b(?:\d{4}|\$[\d,]+|[A-Z]{2,})\b', content))
        
        informativeness_score = (
            unique_ratio * 40 +
            substantive_ratio * 40 +
            (10 if has_numbers else 0) +
            (10 if has_specific_data else 0) -
            boilerplate_count * 10
        )
        
        return max(0, min(100, informativeness_score))
        
    def _validate_metadata(self, metadata: Dict[str, Any]) -> float:
        """Validate metadata completeness."""
        required_fields = ["source", "type", "chunk_index"]
        recommended_fields = ["title", "section", "language", "created_at"]
        
        score = 100
        
        # Check required fields
        for field in required_fields:
            if field not in metadata or not metadata[field]:
                score -= 20
                
        # Check recommended fields
        for field in recommended_fields:
            if field not in metadata or not metadata[field]:
                score -= 5
                
        return max(0, score)
        
    def _get_common_issues(self, issues: List[str]) -> List[Tuple[str, int]]:
        """Get most common issues from list."""
        from collections import Counter
        issue_counts = Counter(issues)
        return issue_counts.most_common(5)
        
    def suggest_improvements(self, chunk: Document, metrics: QualityMetrics) -> List[str]:
        """Suggest specific improvements for a chunk."""
        suggestions = metrics.suggestions.copy()
        
        # Add specific suggestions based on metrics
        if metrics.metrics["coherence_score"] < 70:
            suggestions.append("Ensure chunk starts and ends at natural boundaries")
            
        if metrics.metrics["informativeness_score"] < 50:
            suggestions.append("Consider enriching with more context or removing if redundant")
            
        if metrics.metrics["word_count"] < 20:
            suggestions.append("Chunk may be too granular - consider merging with neighbors")
            
        if "table" in chunk.page_content.lower() and "has_tables" not in chunk.metadata:
            suggestions.append("Detected table content - consider using table-specific processing")
            
        return suggestions
