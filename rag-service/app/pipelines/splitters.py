"""Text splitting strategies for document processing using LangChain."""

from typing import List, Dict, Any, Optional, Callable
import re
from datetime import datetime

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
    HTMLHeaderTextSplitter,
    PythonCodeTextSplitter,
    LatexTextSplitter,
    TokenTextSplitter,
    SentenceTransformersTokenTextSplitter,
    TextSplitter,
    CharacterTextSplitter,
    Language
)
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker as LangChainSemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings

from app.core.config import settings
from app.core.logging import get_logger
from app.models.documents import DocumentType
from app.pipelines.smart_splitters import SmartDocumentSplitter, HierarchicalChunker
from app.components.base import BaseComponent

logger = get_logger(__name__)


class LangChainTextSplitter(BaseComponent):
    """Main text splitter using LangChain's built-in splitters."""
    
    def __init__(self):
        """Initialize with embeddings for semantic splitting."""
        super().__init__()
        self.embeddings = None  # Lazy load when needed
        
    def get_embeddings(self):
        """Get embeddings for semantic chunking."""
        if self.embeddings is None:
            # Use HuggingFaceEmbeddings as default
            self.embeddings = HuggingFaceEmbeddings(
                model_name=settings.embedding_model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        return self.embeddings
        
    def get_text_splitter(
        self,
        splitter_type: str = "recursive",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        doc_type: Optional[DocumentType] = None
    ) -> TextSplitter:
        """Get appropriate text splitter based on type and document type."""
        chunk_size = chunk_size or settings.chunk_size
        chunk_overlap = chunk_overlap or settings.chunk_overlap
        
        # Use smart splitter for complex document types
        if splitter_type == "smart" or (doc_type and doc_type in [
            DocumentType.WEB, DocumentType.PDF, DocumentType.DOCX,
            DocumentType.MARKDOWN, DocumentType.CSV, DocumentType.XLSX
        ]):
            return SmartDocumentSplitter(
                embeddings=self.get_embeddings(),
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                use_sentence_splitter=settings.use_sentence_aware_splitting,
                use_dynamic_sizing=settings.use_dynamic_chunk_sizing
            )
            
        # LangChain splitters
        if splitter_type == "semantic":
            return LangChainSemanticChunker(
                embeddings=self.get_embeddings(),
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=95
            )
        elif splitter_type == "markdown":
            return MarkdownTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        elif splitter_type == "python":
            return PythonCodeTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        elif splitter_type == "latex":
            return LatexTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        elif splitter_type == "html":
            return HTMLHeaderTextSplitter(
                headers_to_split_on=[
                    ("h1", "Header 1"),
                    ("h2", "Header 2"),
                    ("h3", "Header 3"),
                    ("h4", "Header 4"),
                ]
            )
        elif splitter_type == "token":
            return TokenTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        elif splitter_type == "sentence_transformers":
            return SentenceTransformersTokenTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        elif splitter_type == "character":
            return CharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separator="\n\n"
            )
        else:  # Default recursive
            return RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
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
            
    def split_documents(
        self,
        documents: List[Document],
        splitter_type: str = "recursive",
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        doc_type: Optional[DocumentType] = None
    ) -> List[Document]:
        """Split documents using appropriate splitter."""
        try:
            logger.info(f"Splitting {len(documents)} documents")
            all_chunks = []
            
            for doc in documents:
                try:
                    # Get the splitter
                    splitter = self.get_text_splitter(
                        splitter_type=splitter_type,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        doc_type=doc_type
                    )
                    
                    # Add document type to metadata if available
                    if doc_type and 'type' not in doc.metadata:
                        doc.metadata['type'] = doc_type.value if hasattr(doc_type, 'value') else str(doc_type)
                    
                    # Check if it's a SmartDocumentSplitter instance
                    if hasattr(splitter, 'split_by_type'):
                        # Use smart splitting
                        logger.error(f"SPLITTER_DEBUG: About to call split_by_type with doc type={type(doc)}, has_page_content={hasattr(doc, 'page_content')}")
                        if hasattr(doc, 'page_content'):
                            logger.error(f"SPLITTER_DEBUG: doc.page_content length = {len(doc.page_content) if doc.page_content else 0}")
                        else:
                            logger.error(f"SPLITTER_DEBUG: doc object = {doc}")
                        chunks = splitter.split_by_type(doc, doc_type=doc_type)
                    else:
                        # Use regular splitting
                        chunks = splitter.split_documents([doc])
                    
                    all_chunks.extend(chunks)
                    
                except Exception as e:
                    logger.warning(f"Failed to split document: {e}. Using fallback splitter.")
                    # Fallback to basic splitting
                    fallback_splitter = self.get_text_splitter("recursive")
                    chunks = fallback_splitter.split_documents([doc])
                    all_chunks.extend(chunks)
            
            logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
            return all_chunks
            
        except Exception as e:
            logger.error(f"Error splitting documents: {e}")
            raise


class SemanticTextSplitter(TextSplitter):
    """Custom semantic text splitter that preserves document structure."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
        **kwargs
    ):
        """Initialize semantic splitter."""
        super().__init__(**kwargs)
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        
        # Default separators optimized for policy documents
        self.separators = separators or [
            # Section breaks
            "\n\n\n",
            "\n\n",
            # Headers
            "\n# ",
            "\n## ",
            "\n### ",
            # List items
            "\n• ",
            "\n- ",
            "\n* ",
            "\n1. ",
            "\n2. ",
            # Sentences
            ". ",
            "! ",
            "? ",
            # Fallback
            "\n",
            " ",
            "",
        ]
    
    @property
    def chunk_size(self) -> int:
        """Get chunk size."""
        return self._chunk_size
    
    @property
    def chunk_overlap(self) -> int:
        """Get chunk overlap."""
        return self._chunk_overlap
        
    def split_text(self, text: str) -> List[str]:
        """Split text into semantic chunks."""
        # Pre-process text to identify important sections
        sections = self._identify_sections(text)
        
        chunks = []
        for section in sections:
            # If section is small enough, keep it whole
            if len(section["content"]) <= self.chunk_size:
                chunks.append(section["content"])
            else:
                # Split large sections
                section_chunks = self._split_section(section["content"])
                
                # Add section context to each chunk
                for chunk in section_chunks:
                    if section["header"]:
                        chunk = f"{section['header']}\n\n{chunk}"
                    chunks.append(chunk)
                    
        return chunks
        
    def _identify_sections(self, text: str) -> List[Dict[str, str]]:
        """Identify logical sections in the text."""
        sections = []
        
        # Pattern for section headers (numeric and alphanumeric)
        header_pattern = r'^(\d+\.?\d*\.?\d*|\w+\.?\d*\.?\d*)\s+(.+?)$'
        
        lines = text.split('\n')
        current_section = {"header": "", "content": ""}
        
        for line in lines:
            # Check if line is a header
            match = re.match(header_pattern, line.strip())
            if match and len(line.strip()) < 100:  # Headers are usually short
                # Save current section if it has content
                if current_section["content"].strip():
                    sections.append(current_section)
                    
                # Start new section
                current_section = {
                    "header": line.strip(),
                    "content": ""
                }
            else:
                # Add to current section
                current_section["content"] += line + "\n"
                
        # Don't forget the last section
        if current_section["content"].strip():
            sections.append(current_section)
            
        # If no sections found, treat whole text as one section
        if not sections:
            sections = [{"header": "", "content": text}]
            
        return sections
        
    def _split_section(self, text: str) -> List[str]:
        """Split a section into chunks."""
        # Use recursive splitting with our separators
        chunks = []
        current_chunk = ""
        
        for separator in self.separators:
            if separator:
                parts = text.split(separator)
                
                for i, part in enumerate(parts):
                    # Add separator back (except for last part)
                    if i < len(parts) - 1:
                        part += separator
                        
                    # Check if adding part exceeds chunk size
                    if len(current_chunk) + len(part) > self.chunk_size:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = part
                    else:
                        current_chunk += part
                        
                # Update text to be remaining content
                text = current_chunk
                current_chunk = ""
                
        # Add any remaining content
        if text.strip():
            chunks.append(text.strip())
            
        # Apply overlap
        if self.chunk_overlap > 0:
            chunks = self._apply_overlap(chunks)
            
        return chunks
        
    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        """Apply overlap between chunks."""
        if len(chunks) <= 1:
            return chunks
            
        overlapped_chunks = []
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                # First chunk - add start of next chunk
                if i + 1 < len(chunks):
                    next_chunk = chunks[i + 1]
                    overlap_text = next_chunk[:self.chunk_overlap]
                    chunk = f"{chunk}\n... {overlap_text}"
                    
            elif i == len(chunks) - 1:
                # Last chunk - add end of previous chunk
                prev_chunk = chunks[i - 1]
                overlap_text = prev_chunk[-self.chunk_overlap:]
                chunk = f"{overlap_text} ...\n{chunk}"
                
            else:
                # Middle chunks - add both overlaps
                prev_chunk = chunks[i - 1]
                next_chunk = chunks[i + 1]
                prev_overlap = prev_chunk[-self.chunk_overlap//2:]
                next_overlap = next_chunk[:self.chunk_overlap//2]
                chunk = f"{prev_overlap} ...\n{chunk}\n... {next_overlap}"
                
            overlapped_chunks.append(chunk)
            
        return overlapped_chunks


class DocumentSplitterFactory:
    """Factory for creating appropriate text splitters."""
    
    @staticmethod
    def create_splitter(
        doc_type: DocumentType,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        use_semantic: bool = True
    ) -> TextSplitter:
        """Create appropriate splitter based on document type."""
        chunk_size = chunk_size or settings.chunk_size
        chunk_overlap = chunk_overlap or settings.chunk_overlap
        
        # Try to use smart document splitter for better quality
        if use_semantic and doc_type in [DocumentType.WEB, DocumentType.PDF, DocumentType.DOCX]:
            try:
                # Return SmartDocumentSplitter instance which has split_by_type method
                return SmartDocumentSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
            except Exception as e:
                logger.warning(f"Failed to create smart splitter: {e}. Falling back to standard splitter.")
        
        if doc_type == DocumentType.MARKDOWN:
            return MarkdownTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            
        elif doc_type == DocumentType.WEB:
            # Use custom semantic splitter for web content
            return SemanticTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            
        elif doc_type == DocumentType.CSV or doc_type == DocumentType.XLSX:
            # For tabular data, use larger chunks to preserve row context
            return RecursiveCharacterTextSplitter(
                chunk_size=chunk_size * 2,  # Larger chunks for tables
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", " | ", ", ", " ", ""],
            )
            
        else:
            # Default recursive splitter
            return RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
            

def split_documents_table_aware(
    documents: List[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    preserve_tables: bool = True
) -> List[Document]:
    """Split documents with special handling for tables."""
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap
    
    all_chunks = []
    
    for doc in documents:
        content_type = doc.metadata.get("content_type", "text")
        
        # Don't split table documents - keep them intact
        if preserve_tables and content_type in ["table_markdown", "table_key_value", "table_json", "table_unstructured"]:
            # Tables are already optimally formatted, just add them
            all_chunks.append(doc)
            continue
            
        # For regular text, use appropriate splitter
        doc_type = doc.metadata.get("type", DocumentType.TEXT)
        splitter = DocumentSplitterFactory.create_splitter(doc_type, chunk_size, chunk_overlap)
        
        # Split the document
        if hasattr(splitter, 'split_by_type'):
            # SmartDocumentSplitter has a different method
            chunks = splitter.split_by_type(doc, doc_type=doc_type)
        else:
            # Regular LangChain splitters
            chunks = splitter.split_documents([doc])
        
        # Add chunk metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "chunk_index": i,
                "chunk_total": len(chunks),
                "original_content_type": content_type
            })
            
        all_chunks.extend(chunks)
    
    logger.info(f"Table-aware splitting: {len(documents)} documents -> {len(all_chunks)} chunks")
    return all_chunks


def split_documents(
    documents: List[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    preserve_metadata: bool = True
) -> List[Document]:
    """Split documents into chunks with metadata preservation."""
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap
    
    all_chunks = []
    
    for doc in documents:
        # Get document type from metadata
        doc_type = doc.metadata.get("type", DocumentType.TEXT)
        
        # Create appropriate splitter
        splitter = DocumentSplitterFactory.create_splitter(
            doc_type,
            chunk_size,
            chunk_overlap
        )
        
        # Split document
        if hasattr(splitter, 'split_by_type'):
            # SmartDocumentSplitter has a different method
            chunks = splitter.split_by_type(doc, doc_type=doc_type)
        else:
            # Regular LangChain splitters
            chunks = splitter.split_documents([doc])
        
        # Enhance chunk metadata
        for i, chunk in enumerate(chunks):
            if preserve_metadata:
                # Preserve original metadata
                chunk.metadata.update(doc.metadata)
                
            # Add chunk-specific metadata
            chunk.metadata.update({
                "chunk_index": i,
                "chunk_total": len(chunks),
                "parent_id": doc.metadata.get("id", ""),
                "chunked_at": datetime.utcnow().isoformat(),
            })
            
            # Extract section info if available
            section_match = re.match(r'^(\d+\.?\d*\.?\d*)\s+(.+?)$', 
                                   chunk.page_content[:100], re.MULTILINE)
            if section_match:
                chunk.metadata["section"] = section_match.group(1)
                chunk.metadata["section_title"] = section_match.group(2).strip()
                
        all_chunks.extend(chunks)
        
    logger.info(f"Split {len(documents)} documents into {len(all_chunks)} chunks")
    return all_chunks


def create_propositions(chunk: Document) -> List[Document]:
    """Extract propositions from a chunk for better retrieval."""
    propositions = []
    text = chunk.page_content
    
    # Split into sentences
    sentences = re.split(r'[.!?]+', text)
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 20:  # Minimum meaningful length
            # Create proposition document
            prop_doc = Document(
                page_content=sentence,
                metadata={
                    **chunk.metadata,
                    "proposition": True,
                    "parent_chunk": chunk.metadata.get("chunk_index", 0),
                }
            )
            propositions.append(prop_doc)
            
    return propositions


def split_documents_hierarchical(
    documents: List[Document],
    chunk_size: Optional[int] = None,
    preserve_structure: bool = True
) -> List[Document]:
    """Split documents using hierarchical chunking to preserve structure."""
    chunk_size = chunk_size or settings.chunk_size
    hierarchical_chunker = HierarchicalChunker(
        max_chunk_size=chunk_size,
        preserve_headers=preserve_structure
    )
    
    all_chunks = []
    
    for doc in documents:
        # Get document type
        doc_type = doc.metadata.get("type", DocumentType.TEXT)
        
        # Use hierarchical chunking for structured documents
        if doc_type in [DocumentType.MARKDOWN, DocumentType.DOCX, DocumentType.WEB]:
            hierarchical_chunks = hierarchical_chunker.chunk_document(doc.page_content)
            
            for i, chunk_data in enumerate(hierarchical_chunks):
                chunk_doc = Document(
                    page_content=chunk_data['text'],
                    metadata={
                        **doc.metadata,
                        **chunk_data['metadata'],
                        "chunk_index": i,
                        "chunk_total": len(hierarchical_chunks),
                        "chunking_method": "hierarchical"
                    }
                )
                all_chunks.append(chunk_doc)
        else:
            # Fall back to regular splitting
            regular_chunks = split_documents([doc], chunk_size=chunk_size)
            all_chunks.extend(regular_chunks)
            
    logger.info(f"Hierarchical splitting: {len(documents)} documents -> {len(all_chunks)} chunks")
    return all_chunks