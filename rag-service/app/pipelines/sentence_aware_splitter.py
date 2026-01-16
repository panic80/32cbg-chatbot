"""Sentence-aware text splitting with dynamic sizing."""

import re
from typing import List, Dict, Any, Optional, Tuple
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import logging

from langchain_text_splitters import TextSplitter
from langchain_core.documents import Document

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)
    nltk.download('punkt', quiet=True)


class SentenceAwareTextSplitter(TextSplitter):
    """Text splitter that respects sentence boundaries and uses dynamic sizing."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 200,
        max_chunk_size: int = 1500,
        length_function: callable = len,
        respect_sentence_boundary: bool = True,
        adaptive_sizing: bool = True
    ):
        """Initialize sentence-aware splitter.
        
        Args:
            chunk_size: Target chunk size
            chunk_overlap: Overlap between chunks
            min_chunk_size: Minimum allowed chunk size
            max_chunk_size: Maximum allowed chunk size
            length_function: Function to calculate text length
            respect_sentence_boundary: Whether to respect sentence boundaries
            adaptive_sizing: Whether to use adaptive chunk sizing
        """
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function
        )
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.respect_sentence_boundary = respect_sentence_boundary
        self.adaptive_sizing = adaptive_sizing
        
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks respecting sentence boundaries."""
        if not self.respect_sentence_boundary:
            # Fall back to character-based splitting
            return self._split_text_basic(text)
            
        # Split into sentences
        sentences = self._split_into_sentences(text)
        
        if not sentences:
            return []
            
        # Group sentences into chunks
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = self._length_function(sentence)
            
            # Handle very long sentences
            if sentence_size > self.max_chunk_size:
                # Finish current chunk
                if current_chunk:
                    chunks.append(self._join_sentences(current_chunk))
                    current_chunk = []
                    current_size = 0
                
                # Split the long sentence
                sub_chunks = self._split_long_sentence(sentence)
                chunks.extend(sub_chunks)
                continue
                
            # Check if adding this sentence would exceed chunk size
            if current_size + sentence_size > self._chunk_size:
                # Save current chunk if it meets minimum size
                if current_size >= self.min_chunk_size:
                    chunks.append(self._join_sentences(current_chunk))
                    
                    # Start new chunk with overlap
                    if self._chunk_overlap > 0:
                        overlap_sentences = self._get_overlap_sentences(current_chunk)
                        current_chunk = overlap_sentences
                        current_size = sum(self._length_function(s) for s in overlap_sentences)
                    else:
                        current_chunk = []
                        current_size = 0
                        
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_size += sentence_size
            
        # Add final chunk
        if current_chunk and current_size >= self.min_chunk_size:
            chunks.append(self._join_sentences(current_chunk))
            
        # Apply adaptive sizing if enabled
        if self.adaptive_sizing:
            chunks = self._apply_adaptive_sizing(chunks)
            
        return chunks
        
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using NLTK."""
        try:
            # Use NLTK sentence tokenizer
            sentences = sent_tokenize(text)
            
            # Post-process to handle edge cases
            processed_sentences = []
            for sentence in sentences:
                # Handle bullet points and numbered lists
                if re.match(r'^\s*[-•*]\s+', sentence) or re.match(r'^\s*\d+\.\s+', sentence):
                    # Keep list items together if they're short
                    if self._length_function(sentence) < self._chunk_size // 3:
                        if processed_sentences and self._length_function(processed_sentences[-1]) < self._chunk_size // 2:
                            processed_sentences[-1] += '\n' + sentence
                            continue
                            
                processed_sentences.append(sentence)
                
            return processed_sentences
            
        except Exception as e:
            logger.warning(f"NLTK sentence tokenization failed: {e}. Using fallback.")
            # Fallback to simple sentence splitting
            return self._simple_sentence_split(text)
            
    def _simple_sentence_split(self, text: str) -> List[str]:
        """Simple sentence splitting as fallback."""
        # Split on common sentence endings
        sentence_endings = r'[.!?]+[\s\n]+'
        sentences = re.split(sentence_endings, text)
        
        # Clean up and filter
        return [s.strip() for s in sentences if s.strip()]
        
    def _split_long_sentence(self, sentence: str) -> List[str]:
        """Split a very long sentence into smaller chunks."""
        # Try to split on semicolons, commas, or other natural breaks
        parts = []
        
        # First try semicolons
        if ';' in sentence:
            parts = sentence.split(';')
        # Then try commas if we still have long parts
        elif ',' in sentence and self._length_function(sentence) > self.max_chunk_size:
            parts = sentence.split(',')
        # Finally, fall back to word-based splitting
        else:
            words = sentence.split()
            current_part = []
            current_size = 0
            
            for word in words:
                word_size = self._length_function(word) + 1  # +1 for space
                if current_size + word_size > self._chunk_size:
                    if current_part:
                        parts.append(' '.join(current_part))
                    current_part = [word]
                    current_size = word_size
                else:
                    current_part.append(word)
                    current_size += word_size
                    
            if current_part:
                parts.append(' '.join(current_part))
                
        return [p.strip() for p in parts if p.strip()]
        
    def _join_sentences(self, sentences: List[str]) -> str:
        """Join sentences with appropriate spacing."""
        return ' '.join(s.strip() for s in sentences)
        
    def _get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """Get sentences for chunk overlap."""
        if not sentences:
            return []
            
        overlap_size = 0
        overlap_sentences = []
        
        # Add sentences from the end until we reach overlap size
        for sentence in reversed(sentences):
            sentence_size = self._length_function(sentence)
            if overlap_size + sentence_size <= self._chunk_overlap:
                overlap_sentences.insert(0, sentence)
                overlap_size += sentence_size
            else:
                break
                
        return overlap_sentences
        
    def _apply_adaptive_sizing(self, chunks: List[str]) -> List[str]:
        """Apply adaptive sizing based on content density."""
        adapted_chunks = []
        
        for chunk in chunks:
            # Calculate content density metrics
            density = self._calculate_content_density(chunk)
            
            # Adjust chunk based on density
            if density['is_sparse'] and len(chunk) < self._chunk_size * 0.7:
                # Try to merge with next chunk if sparse
                if adapted_chunks and len(adapted_chunks[-1]) + len(chunk) <= self.max_chunk_size:
                    adapted_chunks[-1] += '\n\n' + chunk
                    continue
                    
            elif density['is_dense'] and len(chunk) > self._chunk_size * 1.3:
                # Split dense chunks further
                sub_chunks = self._split_dense_content(chunk, density)
                adapted_chunks.extend(sub_chunks)
                continue
                
            adapted_chunks.append(chunk)
            
        return adapted_chunks
        
    def _calculate_content_density(self, text: str) -> Dict[str, Any]:
        """Calculate content density metrics."""
        words = word_tokenize(text.lower())
        unique_words = set(words)
        
        # Calculate metrics
        total_words = len(words)
        unique_ratio = len(unique_words) / total_words if total_words > 0 else 0
        avg_word_length = sum(len(w) for w in words) / total_words if total_words > 0 else 0
        
        # Check for special content
        has_numbers = bool(re.search(r'\d+', text))
        has_special_chars = bool(re.search(r'[^a-zA-Z0-9\s.,!?]', text))
        line_count = len(text.split('\n'))
        
        # Determine density
        is_dense = (
            unique_ratio > 0.7 or  # High vocabulary diversity
            avg_word_length > 6 or  # Long words
            has_special_chars or    # Technical content
            (has_numbers and total_words > 50)  # Numerical data
        )
        
        is_sparse = (
            unique_ratio < 0.3 or  # Repetitive content
            line_count > total_words / 10  # Many short lines
        )
        
        return {
            'total_words': total_words,
            'unique_ratio': unique_ratio,
            'avg_word_length': avg_word_length,
            'has_numbers': has_numbers,
            'has_special_chars': has_special_chars,
            'is_dense': is_dense,
            'is_sparse': is_sparse
        }
        
    def _split_dense_content(self, text: str, density: Dict[str, Any]) -> List[str]:
        """Split dense content into smaller chunks."""
        # Use smaller chunk size for dense content
        target_size = int(self._chunk_size * 0.7)
        
        # Split into paragraphs first
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para_size = self._length_function(para)
            
            if current_size + para_size > target_size and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_size = para_size
            else:
                current_chunk.append(para)
                current_size += para_size
                
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            
        return chunks
        
    def _split_text_basic(self, text: str) -> List[str]:
        """Basic text splitting without sentence awareness."""
        # Fall back to simple character-based splitting
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self._chunk_size
            
            # Try to find a good break point
            if end < len(text):
                # Look for paragraph break
                para_break = text.rfind('\n\n', start, end)
                if para_break > start:
                    end = para_break
                else:
                    # Look for sentence end
                    for sep in ['. ', '! ', '? ', '\n']:
                        pos = text.rfind(sep, start, end)
                        if pos > start:
                            end = pos + len(sep) - 1
                            break
                            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
                
            # Calculate overlap start
            if self._chunk_overlap > 0 and end < len(text):
                start = max(start + 1, end - self._chunk_overlap)
            else:
                start = end
                
        return chunks


class DynamicChunkSizer:
    """Determines optimal chunk size based on content analysis."""
    
    def __init__(self, base_chunk_size: int = 1024):
        """Initialize dynamic chunk sizer."""
        self.base_chunk_size = base_chunk_size
        
    def calculate_optimal_size(
        self,
        text: str,
        doc_type: str,
        metadata: Dict[str, Any]
    ) -> Tuple[int, int]:  # (chunk_size, chunk_overlap)
        """Calculate optimal chunk size and overlap for given content."""
        # Analyze content characteristics
        analysis = self._analyze_content(text)
        
        # Base adjustments by document type
        type_factors = {
            "pdf": 1.0,
            "web": 0.9,      # Slightly smaller for web content
            "markdown": 0.85, # Smaller to preserve structure
            "csv": 2.0,      # Much larger for tabular data
            "xlsx": 2.0,
            "docx": 1.1,     # Slightly larger for documents
            "text": 1.0
        }
        
        type_factor = type_factors.get(doc_type, 1.0)
        
        # Adjust based on content analysis
        if analysis['avg_paragraph_length'] > 500:
            # Long paragraphs - use larger chunks
            content_factor = 1.3
        elif analysis['avg_paragraph_length'] < 100:
            # Short paragraphs - use smaller chunks
            content_factor = 0.7
        else:
            content_factor = 1.0
            
        # Adjust for technical content
        if analysis['is_technical']:
            technical_factor = 1.2  # Larger chunks for context
        else:
            technical_factor = 1.0
            
        # Calculate final size
        optimal_size = int(self.base_chunk_size * type_factor * content_factor * technical_factor)
        
        # Ensure within bounds
        optimal_size = max(200, min(optimal_size, 2000))
        
        # Calculate overlap (typically 10-20% of chunk size)
        if analysis['has_lists'] or analysis['is_technical']:
            overlap = int(optimal_size * 0.2)  # More overlap for structured content
        else:
            overlap = int(optimal_size * 0.15)
            
        return optimal_size, overlap
        
    def _analyze_content(self, text: str) -> Dict[str, Any]:
        """Analyze content characteristics."""
        # Split into paragraphs
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        
        # Calculate metrics
        if paragraphs:
            avg_paragraph_length = sum(len(p) for p in paragraphs) / len(paragraphs)
        else:
            avg_paragraph_length = len(text)
            
        # Check for technical indicators
        technical_patterns = [
            r'\b(?:function|class|def|import|export|const|let|var)\b',
            r'\b(?:SELECT|FROM|WHERE|INSERT|UPDATE)\b',
            r'[<>{}()\[\]]',
            r'\b\d+\.\d+\.\d+\b',  # Version numbers
        ]
        
        is_technical = any(re.search(pattern, text[:1000]) for pattern in technical_patterns)
        
        # Check for lists
        has_lists = bool(re.search(r'^\s*[-•*\d]+\.?\s+', text, re.MULTILINE))
        
        # Check for tables
        has_tables = '|' in text and text.count('|') > 10
        
        return {
            'avg_paragraph_length': avg_paragraph_length,
            'is_technical': is_technical,
            'has_lists': has_lists,
            'has_tables': has_tables,
            'paragraph_count': len(paragraphs)
        }