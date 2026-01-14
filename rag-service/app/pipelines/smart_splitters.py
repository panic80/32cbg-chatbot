"""Smart text splitting strategies using LangChain for document processing."""

from typing import List, Dict, Any, Optional, Tuple
import re
from datetime import datetime

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    HTMLHeaderTextSplitter,
    Language,
    CharacterTextSplitter
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.core.config import settings
from app.core.logging import get_logger
from app.models.documents import DocumentType
from app.core.langchain_config import get_embeddings
from app.pipelines.sentence_aware_splitter import (
    SentenceAwareTextSplitter, DynamicChunkSizer
)

logger = get_logger(__name__)


class SmartDocumentSplitter:
    """Document-type aware splitting strategy using LangChain splitters."""
    
    def __init__(
        self,
        embeddings: Optional[Embeddings] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        use_sentence_splitter: bool = True,
        use_dynamic_sizing: bool = True
    ):
        """Initialize smart splitter with embeddings for semantic chunking."""
        self.embeddings = embeddings or get_embeddings()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_sentence_splitter = use_sentence_splitter
        self.use_dynamic_sizing = use_dynamic_sizing
        
        # Initialize dynamic chunk sizer
        self.chunk_sizer = DynamicChunkSizer(base_chunk_size=chunk_size)
        
        # Initialize splitters for different content types
        self._init_splitters()
        
    def _init_splitters(self):
        """Initialize various splitters for different document types."""
        # Semantic chunker for narrative text
        self.semantic_chunker = SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95,
            number_of_chunks=None  # Let it determine optimal chunks
        )
        
        # Markdown splitter with header preservation
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
                ("####", "Header 4"),
            ],
            return_each_line=False,
            strip_headers=False
        )
        
        # HTML splitter for web content
        self.html_splitter = HTMLHeaderTextSplitter(
            headers_to_split_on=[
                ("h1", "Header 1"),
                ("h2", "Header 2"),
                ("h3", "Header 3"),
                ("h4", "Header 4"),
            ]
        )
        
        # Code splitter for technical documents
        self.code_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,  # Can be adapted based on content
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        # Sentence-aware splitter
        if self.use_sentence_splitter:
            self.sentence_splitter = SentenceAwareTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                min_chunk_size=200,
                max_chunk_size=1500,
                respect_sentence_boundary=True,
                adaptive_sizing=self.use_dynamic_sizing
            )
        
        # Default recursive splitter
        self.default_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[
                "\n\n\n",    # Multiple newlines
                "\n\n",      # Paragraphs
                "\n",        # Lines
                ". ",        # Sentences
                ", ",        # Clauses
                " ",         # Words
                ""           # Characters
            ],
            keep_separator=True,
            is_separator_regex=False
        )
        
    def detect_document_structure(self, text: str) -> Dict[str, Any]:
        """Detect document structure and content type."""
        structure = {
            "has_headers": False,
            "has_tables": False,
            "has_code": False,
            "has_lists": False,
            "primary_type": "narrative",
            "header_count": 0,
            "table_count": 0,
            "list_count": 0
        }

        # Check for markdown headers
        header_pattern = r'^#{1,6}\s+.+$'
        headers = re.findall(header_pattern, text, re.MULTILINE)
        structure["header_count"] = len(headers)
        structure["has_headers"] = len(headers) > 0

        # Check for tables (markdown or HTML)
        table_patterns = [
            r'\|.*\|.*\|',  # Markdown tables
            r'<table[^>]*>',  # HTML tables
            r'┌─.*─┐',  # ASCII tables
        ]
        for pattern in table_patterns:
            if re.search(pattern, text):
                structure["has_tables"] = True
                structure["table_count"] += len(re.findall(pattern, text))

        # Check for code blocks
        code_patterns = [
            r'```[\s\S]*?```',  # Markdown code blocks
            r'<code>[\s\S]*?</code>',  # HTML code
        ]
        for pattern in code_patterns:
            if re.search(pattern, text):
                structure["has_code"] = True

        # Check for lists
        list_patterns = [
            r'^\s*[-*+]\s+',  # Unordered lists
            r'^\s*\d+\.\s+',  # Ordered lists
        ]
        for pattern in list_patterns:
            lists = re.findall(pattern, text, re.MULTILINE)
            structure["list_count"] += len(lists)
        structure["has_lists"] = structure["list_count"] > 0

        # Determine primary type
        if structure["has_tables"] and structure["table_count"] > 5:
            structure["primary_type"] = "tabular"
        elif structure["has_headers"] and structure["header_count"] > 10:
            structure["primary_type"] = "structured"
        elif structure["has_code"]:
            structure["primary_type"] = "technical"
        else:
            structure["primary_type"] = "narrative"

        return structure

    def _find_section_context(
        self,
        chunk_text: str,
        structure_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Find which section/chapter this chunk belongs to based on structure_info.
        Returns section context with chapter, section, paragraph numbers.
        """
        context = {
            "chapter": None,
            "section": None,
            "section_title": None,
            "paragraph": None,
            "section_hierarchy": []
        }

        if not structure_info or not structure_info.get("has_structure"):
            return context

        # Get first few lines of chunk to determine context
        first_lines = chunk_text[:500]

        # Check for chapter
        chapters = structure_info.get("chapters", [])
        for chapter in chapters:
            chapter_num = chapter.get("number")
            if chapter_num and (
                f"CHAPTER {chapter_num}" in first_lines.upper() or
                f"Chapter {chapter_num}" in first_lines or
                f"Ch. {chapter_num}" in first_lines
            ):
                context["chapter"] = chapter_num
                break

        # Check for section
        sections = structure_info.get("sections", [])
        for section in sections:
            section_num = section.get("number")
            section_title = section.get("title")
            if section_num and section_num in first_lines:
                context["section"] = section_num
                context["section_title"] = section_title
                # Build hierarchy
                context["section_hierarchy"].append(f"Section {section_num}")
                if section_title:
                    context["section_hierarchy"][-1] += f" ({section_title})"
                break

        # Check for paragraph
        paragraphs = structure_info.get("paragraphs", [])
        for para in paragraphs:
            para_num = para.get("number")
            if para_num and para_num in first_lines:
                context["paragraph"] = para_num
                break

        return context

    def _build_section_hierarchy(
        self,
        context: Dict[str, Any]
    ) -> List[str]:
        """
        Build hierarchical path for citation.
        E.g., ["Chapter 5", "Section 5.2", "Paragraph 5.2.3"]
        """
        hierarchy = []

        if context.get("chapter"):
            hierarchy.append(f"Chapter {context['chapter']}")

        if context.get("section"):
            section_label = f"Section {context['section']}"
            if context.get("section_title"):
                section_label += f" ({context['section_title']})"
            hierarchy.append(section_label)

        if context.get("paragraph"):
            hierarchy.append(f"Paragraph {context['paragraph']}")

        return hierarchy
        
    def split_by_type(
        self,
        document: Document,
        doc_type: Optional[DocumentType] = None
    ) -> List[Document]:
        """Split document based on its type and structure."""
        # Handle case where a list is passed instead of a single document
        if isinstance(document, list):
            logger.error(f"CRITICAL BUG: split_by_type received a list instead of a Document! Processing each item individually.")
            all_chunks = []
            for doc in document:
                chunks = self.split_by_type(doc, doc_type)
                all_chunks.extend(chunks)
            return all_chunks
            
        text = document.page_content
        metadata = document.metadata
        
        # Detect structure if not provided
        structure = self.detect_document_structure(text)
        
        # Override doc_type based on metadata if available
        if not doc_type and "type" in metadata:
            doc_type = metadata["type"]
            
        # Choose appropriate splitting strategy
        if doc_type == DocumentType.MARKDOWN or structure["primary_type"] == "structured":
            return self._split_structured_document(document, structure)
        elif doc_type == DocumentType.CSV or doc_type == DocumentType.XLSX or structure["primary_type"] == "tabular":
            return self._split_tabular_document(document, structure)
        elif structure["primary_type"] == "technical":
            return self._split_technical_document(document, structure)
        else:
            return self._split_narrative_document(document, structure)
            
    def _split_structured_document(
        self,
        document: Document,
        structure: Dict[str, Any]
    ) -> List[Document]:
        """Split structured documents preserving hierarchy and section context."""
        try:
            # Extract structure_info from metadata (from Phase 2 CLI extraction)
            structure_info = document.metadata.get("structure_info")

            # First split by headers to preserve structure
            header_chunks = self.markdown_splitter.split_text(document.page_content)

            # Then apply size-based splitting if needed, but try to respect section boundaries
            final_chunks = []
            for header_doc in header_chunks:
                if len(header_doc.page_content) > self.chunk_size * 1.5:
                    # Further split large sections
                    # Use paragraph boundaries as separators to avoid mid-paragraph splits
                    section_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=self.chunk_size,
                        chunk_overlap=self.chunk_overlap,
                        separators=[
                            "\n\n\n",    # Multiple paragraph breaks
                            "\n\n",      # Paragraph breaks
                            "\n",        # Line breaks
                            ". ",        # Sentence boundaries
                            " ",         # Word boundaries
                            ""           # Characters
                        ],
                        keep_separator=True
                    )
                    sub_chunks = section_splitter.split_documents([header_doc])
                    final_chunks.extend(sub_chunks)
                else:
                    final_chunks.append(header_doc)

            # Enhance metadata with section context
            for i, chunk in enumerate(final_chunks):
                # Find section context for this chunk
                section_context = {}
                if structure_info:
                    section_context = self._find_section_context(
                        chunk.page_content,
                        structure_info
                    )

                # Build section hierarchy
                section_hierarchy = self._build_section_hierarchy(section_context)

                # Update chunk metadata
                chunk.metadata.update({
                    **document.metadata,
                    "chunk_index": i,
                    "chunk_total": len(final_chunks),
                    "splitting_method": "structured_with_context",
                    "document_structure": structure,
                    # Add section context for better citations
                    "chapter": section_context.get("chapter"),
                    "section": section_context.get("section"),
                    "section_title": section_context.get("section_title"),
                    "paragraph": section_context.get("paragraph"),
                    "section_hierarchy": section_hierarchy,
                })

            return final_chunks

        except Exception as e:
            logger.warning(f"Structured splitting failed, falling back to default: {e}")
            return self._split_narrative_document(document, structure)
            
    def _split_tabular_document(
        self,
        document: Document,
        structure: Dict[str, Any]
    ) -> List[Document]:
        """Split tabular documents preserving table integrity."""
        # For tabular data, use larger chunks to preserve context
        tabular_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size * 2,  # Larger chunks for tables
            chunk_overlap=self.chunk_overlap,
            separators=[
                "\n\n",      # Table breaks
                "\n",        # Row breaks
                " | ",       # Column separators
                ", ",        # CSV separators
                "\t",        # Tab separators
                " ",         # Space
                ""           # Characters
            ],
            keep_separator=True
        )
        
        chunks = tabular_splitter.split_documents([document])
        
        # Enhance metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                **document.metadata,
                "chunk_index": i,
                "chunk_total": len(chunks),
                "splitting_method": "tabular",
                "document_structure": structure
            })
            
        return chunks
        
    def _split_technical_document(
        self,
        document: Document,
        structure: Dict[str, Any]
    ) -> List[Document]:
        """Split technical documents preserving code blocks."""
        # Extract code blocks first
        code_blocks = re.findall(r'```[\s\S]*?```', document.page_content)
        
        # Replace code blocks with placeholders
        text_with_placeholders = document.page_content
        for i, code_block in enumerate(code_blocks):
            text_with_placeholders = text_with_placeholders.replace(
                code_block,
                f"<CODE_BLOCK_{i}>"
            )
            
        # Split the text
        chunks = self.default_splitter.split_text(text_with_placeholders)
        
        # Restore code blocks
        final_chunks = []
        for chunk_text in chunks:
            # Restore any code blocks in this chunk
            for i, code_block in enumerate(code_blocks):
                chunk_text = chunk_text.replace(f"<CODE_BLOCK_{i}>", code_block)
                
            chunk_doc = Document(
                page_content=chunk_text,
                metadata={
                    **document.metadata,
                    "splitting_method": "technical",
                    "document_structure": structure
                }
            )
            final_chunks.append(chunk_doc)
            
        # Update metadata
        for i, chunk in enumerate(final_chunks):
            chunk.metadata.update({
                "chunk_index": i,
                "chunk_total": len(final_chunks)
            })
            
        return final_chunks
        
    def _split_narrative_document(
        self,
        document: Document,
        structure: Dict[str, Any]
    ) -> List[Document]:
        """Split narrative documents using sentence-aware or semantic chunking."""
        splitting_method = "narrative"
        
        # Calculate optimal chunk size if dynamic sizing is enabled
        if self.use_dynamic_sizing:
            doc_type = document.metadata.get("type", "text")
            optimal_size, optimal_overlap = self.chunk_sizer.calculate_optimal_size(
                document.page_content, doc_type, document.metadata
            )
            logger.debug(f"Dynamic sizing: chunk_size={optimal_size}, overlap={optimal_overlap}")
        else:
            optimal_size = self.chunk_size
            optimal_overlap = self.chunk_overlap
        
        try:
            if self.use_sentence_splitter and hasattr(self, 'sentence_splitter'):
                # Update sentence splitter with optimal size
                self.sentence_splitter._chunk_size = optimal_size
                self.sentence_splitter._chunk_overlap = optimal_overlap
                
                # Use sentence-aware splitting
                chunks = self.sentence_splitter.split_documents([document])
                splitting_method = "sentence_aware"
            else:
                # Use semantic chunking for better context preservation
                chunks = self.semantic_chunker.split_documents([document])
                splitting_method = "semantic"
                
                # If semantic chunking produces too few or too many chunks, fall back
                if len(chunks) < 2 or len(chunks) > len(document.page_content) // 200:
                    raise ValueError("Semantic chunking produced suboptimal results")
                    
        except Exception as e:
            logger.debug(f"Advanced chunking failed, using default splitter: {e}")
            # Update default splitter with optimal size
            self.default_splitter._chunk_size = optimal_size
            self.default_splitter._chunk_overlap = optimal_overlap
            chunks = self.default_splitter.split_documents([document])
            splitting_method = "default"
            
        # Enhance metadata with section context if available
        structure_info = document.metadata.get("structure_info")

        for i, chunk in enumerate(chunks):
            # Find section context for this chunk if structure_info exists
            section_context = {}
            section_hierarchy = []

            if structure_info and structure_info.get("has_structure"):
                section_context = self._find_section_context(
                    chunk.page_content,
                    structure_info
                )
                section_hierarchy = self._build_section_hierarchy(section_context)

            chunk.metadata.update({
                **document.metadata,
                "chunk_index": i,
                "chunk_total": len(chunks),
                "splitting_method": splitting_method,
                "document_structure": structure,
                "chunk_size_used": optimal_size,
                "chunk_overlap_used": optimal_overlap,
                # Add section context if detected
                "chapter": section_context.get("chapter"),
                "section": section_context.get("section"),
                "section_title": section_context.get("section_title"),
                "paragraph": section_context.get("paragraph"),
                "section_hierarchy": section_hierarchy,
            })

        return chunks
        
    def optimize_chunk_size(
        self,
        content_type: str,
        base_chunk_size: int
    ) -> int:
        """Optimize chunk size based on content type."""
        optimization_factors = {
            "narrative": 1.0,      # Standard size
            "structured": 0.8,     # Smaller for better section preservation
            "tabular": 2.0,        # Larger to keep tables intact
            "technical": 1.2,      # Slightly larger for code context
            "table_markdown": 3.0, # Much larger for complex tables
            "table_html": 3.0,
            "table_key_value": 1.5
        }
        
        factor = optimization_factors.get(content_type, 1.0)
        return int(base_chunk_size * factor)


class HierarchicalChunker:
    """Hierarchical document chunking that preserves document structure."""
    
    def __init__(
        self,
        max_chunk_size: int = 1000,
        min_chunk_size: int = 100,
        preserve_headers: bool = True
    ):
        """Initialize hierarchical chunker."""
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.preserve_headers = preserve_headers
        
    def chunk_document(self, text: str) -> List[Dict[str, Any]]:
        """Chunk document hierarchically preserving structure."""
        # Parse document structure
        sections = self._parse_document_hierarchy(text)
        
        # Chunk each section appropriately
        chunks = []
        for section in sections:
            section_chunks = self._chunk_section(section)
            chunks.extend(section_chunks)
            
        return chunks
        
    def _parse_document_hierarchy(self, text: str) -> List[Dict[str, Any]]:
        """Parse document into hierarchical sections."""
        sections = []
        current_section = {
            "level": 0,
            "title": "Document",
            "content": "",
            "subsections": []
        }
        
        # Simple parsing - can be enhanced
        lines = text.split('\n')
        for line in lines:
            # Check if it's a header
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if header_match:
                level = len(header_match.group(1))
                title = header_match.group(2)
                
                # Save current section if it has content
                if current_section["content"].strip():
                    sections.append(current_section)
                    
                # Start new section
                current_section = {
                    "level": level,
                    "title": title,
                    "content": "",
                    "subsections": []
                }
            else:
                current_section["content"] += line + "\n"
                
        # Don't forget the last section
        if current_section["content"].strip():
            sections.append(current_section)
            
        return sections
        
    def _chunk_section(self, section: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk a section respecting size constraints."""
        chunks = []
        content = section["content"].strip()
        
        if len(content) <= self.max_chunk_size:
            # Section fits in one chunk
            chunks.append({
                "text": content,
                "metadata": {
                    "section_level": section["level"],
                    "section_title": section["title"],
                    "is_complete_section": True
                }
            })
        else:
            # Need to split section
            parts = self._split_content(content)
            for i, part in enumerate(parts):
                chunk_text = part
                if self.preserve_headers and i > 0:
                    # Add section header to continuation chunks
                    chunk_text = f"{section['title']} (continued)\n\n{part}"
                    
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "section_level": section["level"],
                        "section_title": section["title"],
                        "is_complete_section": False,
                        "part_index": i,
                        "total_parts": len(parts)
                    }
                })
                
        return chunks
        
    def _split_content(self, content: str) -> List[str]:
        """Split content into appropriately sized chunks."""
        # Simple splitting - can be enhanced with better logic
        words = content.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            word_size = len(word) + 1  # +1 for space
            if current_size + word_size > self.max_chunk_size and current_size >= self.min_chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_size = word_size
            else:
                current_chunk.append(word)
                current_size += word_size
                
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks