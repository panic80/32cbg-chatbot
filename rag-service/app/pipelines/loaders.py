"""Document loaders for various content types using LangChain."""

import asyncio
import aiohttp
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse
import re
import os
from pathlib import Path

from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader,
    UnstructuredURLLoader,
    UnstructuredFileLoader,
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    UnstructuredCSVLoader,
    UnstructuredPowerPointLoader,
    UnstructuredEmailLoader,
    UnstructuredImageLoader,
    UnstructuredEPubLoader,
    UnstructuredRTFLoader
)
from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader
from langchain_community.document_transformers import Html2TextTransformer
from bs4 import BeautifulSoup, SoupStrainer
import httpx
try:
    from unstructured.partition.html import partition_html
    from unstructured.documents.elements import Table as UnstructuredTable
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False

from app.core.config import settings
from app.core.logging import get_logger
from app.models.documents import DocumentType, DocumentMetadata
from app.components.base import BaseComponent
from app.pipelines.table_aware_loader import TableAwareWebLoader, UnstructuredTableLoader
from app.pipelines.pdf_table_extractor import PDFTableExtractor
from app.utils.table_validator import TableValidator, TableStructure

logger = get_logger(__name__)


class EnhancedWebLoader:
    """Enhanced web loader with better parsing for canada.ca."""
    
    def __init__(self, url: str, timeout: int = 30):
        """Initialize web loader."""
        self.url = url
        self.timeout = timeout
        self.session = None
        
    async def load(self) -> List[Document]:
        """Load and parse web content."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    self.url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (compatible; CFTravelBot/1.0)",
                        "Accept": "text/html,application/xhtml+xml",
                        "Accept-Language": "en-CA,en;q=0.9",
                    }
                )
                response.raise_for_status()
                
            # Parse HTML
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Extract metadata
            metadata = self._extract_metadata(soup)
            
            # Extract main content
            content = self._extract_content(soup)
            
            # Create document
            doc = Document(
                page_content=content,
                metadata={
                    "source": self.url,
                    "type": DocumentType.WEB,
                    "title": metadata.get("title", ""),
                    "last_modified": metadata.get("last_modified"),
                    "scraped_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            
            return [doc]
            
        except Exception as e:
            logger.error(f"Failed to load web content from {self.url}: {e}")
            raise
            
    def _extract_metadata(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract metadata from HTML."""
        metadata = {}
        
        # Extract title
        title_tag = soup.find("title")
        if title_tag:
            metadata["title"] = title_tag.get_text(strip=True)
            
        # Extract last modified date
        date_modified = soup.find("time", {"property": "dateModified"})
        if date_modified:
            metadata["last_modified"] = date_modified.get("datetime")
            
        # Extract description
        meta_desc = soup.find("meta", {"name": "description"})
        if meta_desc:
            metadata["description"] = meta_desc.get("content", "")
            
        return metadata
        
    def _extract_content(self, soup: BeautifulSoup) -> str:
        """Extract and clean main content."""
        # Remove unwanted elements
        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()
            
        # Look for main content areas
        content_areas = [
            soup.find("main"),
            soup.find("article"),
            soup.find(class_="content"),
            soup.find(id="content"),
            soup.find(class_="main-content"),
            soup.find(class_="mwsbodytext"),  # Canada.ca specific
            soup.find(class_="gc-cnt-stts"),  # Canada.ca content status
        ]
        
        # Find the first non-empty content area
        content = None
        for area in content_areas:
            if area:
                content = area
                break
                
        if not content:
            content = soup.find("body")
            
        if not content:
            return ""
            
        # Extract text with structure preservation
        text_parts = []
        
        for element in content.find_all(["h1", "h2", "h3", "h4", "p", "li", "table"]):
            if element.name.startswith("h"):
                # Add spacing for headers
                text_parts.append(f"\n\n{element.get_text(strip=True)}\n")
            elif element.name == "li":
                # Preserve list structure
                text_parts.append(f"• {element.get_text(strip=True)}")
            elif element.name == "table":
                # Extract table data
                table_text = self._extract_table(element)
                if table_text:
                    text_parts.append(f"\n{table_text}\n")
            else:
                text_parts.append(element.get_text(strip=True))
                
        # Join and clean
        text = "\n".join(text_parts)
        
        # Clean up excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        return text.strip()
        
    def _extract_table(self, table) -> str:
        """Extract table content with validation and structure preservation."""
        try:
            # Extract table title if present
            title = None
            caption = table.find("caption")
            if caption:
                title = caption.get_text(strip=True)
            
            # Look for preceding header
            prev_sibling = table.find_previous_sibling(["h1", "h2", "h3", "h4", "h5", "h6"])
            if prev_sibling and not title:
                title = prev_sibling.get_text(strip=True)
            
            # Extract all rows
            all_rows = []
            
            # First try to get headers from th elements
            header_row = []
            for th in table.find_all("th"):
                header_row.append(th.get_text(strip=True))
            
            if header_row:
                all_rows.append(header_row)
            
            # Extract data rows
            for tr in table.find_all("tr"):
                cells = []
                # Check both td and th elements (some tables use th in data rows)
                for cell in tr.find_all(["td", "th"]):
                    cells.append(cell.get_text(strip=True))
                if cells and cells != header_row:  # Avoid duplicate header row
                    all_rows.append(cells)
            
            if not all_rows:
                return ""
            
            # Validate and structure the table
            table_structure = TableValidator.validate_table_structure(all_rows)
            if title:
                table_structure.title = title
            
            # Generate enhanced markdown representation
            return table_structure.to_markdown()
            
        except Exception as e:
            logger.warning(f"Error extracting table: {e}")
            # Fallback to simple extraction
            rows = []
            for tr in table.find_all("tr"):
                cells = []
                for cell in tr.find_all(["td", "th"]):
                    cells.append(cell.get_text(strip=True))
                if cells:
                    rows.append(" | ".join(cells))
            return "\n".join(rows)


class LangChainDocumentLoader(BaseComponent):
    """Document loader using LangChain's built-in loaders with enhanced metadata."""
    
    def __init__(self):
        super().__init__()
        self.table_loader = UnstructuredTableLoader()
        self.html_transformer = Html2TextTransformer()
        self._table_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._table_cache: Dict[str, List[Dict[str, Any]]] = {}
        
        # Mapping of file extensions to LangChain loaders
        self.loader_mapping = {
            ".pdf": PyPDFLoader,
            ".doc": UnstructuredWordDocumentLoader,
            ".docx": UnstructuredWordDocumentLoader,
            ".xls": UnstructuredExcelLoader,
            ".xlsx": UnstructuredExcelLoader,
            ".csv": UnstructuredCSVLoader,
            ".ppt": UnstructuredPowerPointLoader,
            ".pptx": UnstructuredPowerPointLoader,
            ".md": UnstructuredMarkdownLoader,
            ".html": UnstructuredHTMLLoader,
            ".htm": UnstructuredHTMLLoader,
            ".txt": TextLoader,
            ".rtf": UnstructuredRTFLoader,
            ".epub": UnstructuredEPubLoader,
            ".eml": UnstructuredEmailLoader,
            ".msg": UnstructuredEmailLoader,
            ".png": UnstructuredImageLoader,
            ".jpg": UnstructuredImageLoader,
            ".jpeg": UnstructuredImageLoader,
            ".gif": UnstructuredImageLoader,
            ".bmp": UnstructuredImageLoader,
        }
    
    def _get_loader_for_file(self, file_path: str) -> Optional[BaseLoader]:
        """Get the appropriate LangChain loader for a file."""
        file_ext = Path(file_path).suffix.lower()
        loader_class = self.loader_mapping.get(file_ext)
        
        if loader_class:
            try:
                # Configure loader with appropriate settings
                loader_kwargs = {}
                
                if loader_class == TextLoader:
                    loader_kwargs = {
                        "file_path": file_path,
                        "encoding": "utf-8"
                    }
                elif loader_class == PyPDFLoader:
                    loader_kwargs = {
                        "file_path": file_path
                    }
                else:
                    loader_kwargs = {
                        "file_path": file_path,
                        "mode": "single",
                        "strategy": "auto"
                    }
                
                return loader_class(**loader_kwargs)
            except Exception as e:
                logger.warning(f"Failed to create {loader_class.__name__}: {e}. Falling back to UnstructuredFileLoader.")
                return UnstructuredFileLoader(file_path, mode="single", strategy="auto")
        else:
            # Default to generic UnstructuredFileLoader
            return UnstructuredFileLoader(file_path, mode="single", strategy="auto")
    
    async def load_from_file(self, file_path: str) -> List[Document]:
        """Load documents from a file using appropriate LangChain loader."""
        try:
            loader = self._get_loader_for_file(file_path)
            
            # Load documents asynchronously if possible
            if hasattr(loader, 'aload'):
                documents = await loader.aload()
            else:
                # Run in executor for sync loaders
                loop = asyncio.get_event_loop()
                documents = await loop.run_in_executor(None, loader.load)

            # Enhance metadata
            file_name = Path(file_path).name
            file_ext = Path(file_path).suffix.lower()
            if not documents:
                return []

            for doc in documents:
                if not doc.metadata:
                    doc.metadata = {}

                doc.metadata.update({
                    "source": file_path,
                    "source_type": "file",
                    "file_name": file_name,
                    "file_extension": file_ext,
                    "document_type": self._detect_document_type(file_ext),
                    "ingestion_timestamp": datetime.now(timezone.utc).isoformat()
                })
            tables: List[Dict[str, Any]] = []
            if file_ext == ".pdf":
                tables = await self._extract_pdf_tables(file_path)
            elif file_ext in [".html", ".htm"]:
                tables = await self._extract_html_tables(file_path)

            if tables:
                table_docs = self._create_table_documents(documents[0].metadata, tables)
                for doc in documents:
                    doc.metadata["has_tables"] = True
                    doc.metadata["table_count"] = len(tables)
                documents.extend(table_docs)

            return documents
            
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}", exc_info=True)
            return []
    
    async def load_from_url(self, url: str, extract_tables: bool = True) -> List[Document]:
        """Load documents from a URL with enhanced extraction."""
        try:
            # Extract URL metadata
            parsed_url = urlparse(url)
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
            
            # Check if it's a government site
            is_gov_site = any(domain in parsed_url.netloc.lower()
                            for domain in ['.gc.ca', '.canada.ca', 'njc-cnm.gc.ca'])
            
            if is_gov_site and extract_tables:
                logger.info(f"Loading government site with table extraction: {url}")
                # Use table-aware loader for government sites
                loader = TableAwareWebLoader(url=url, timeout=30)
                documents = await loader.load()
            else:
                # Use UnstructuredURLLoader for better content extraction
                loader = UnstructuredURLLoader(
                    urls=[url],
                    mode="single",
                    strategy="fast",
                    headers={
                        "User-Agent": "Mozilla/5.0 (compatible; RAGBot/1.0)"
                    }
                )
                
                # Load documents
                loop = asyncio.get_event_loop()
                documents = await loop.run_in_executor(None, loader.load)
                
                # Apply HTML to text transformation if needed
                for doc in documents:
                    if self._is_html_content(doc.page_content):
                        transformed = self.html_transformer.transform_documents([doc])
                        if transformed:
                            doc.page_content = transformed[0].page_content
            
            # Add source metadata
            for doc in documents:
                if not doc.metadata:
                    doc.metadata = {}
                doc.metadata.update({
                    "source": url,
                    "source_type": "web",
                    "base_url": base_url,
                    "is_government_source": is_gov_site,
                    "ingestion_timestamp": datetime.now(timezone.utc).isoformat()
                })
                
            return documents
            
        except Exception as e:
            logger.error(f"Error loading URL {url}: {str(e)}", exc_info=True)
            return []
    
    def _detect_document_type(self, file_ext: str) -> str:
        """Detect document type from file extension."""
        type_mapping = {
            ".pdf": "pdf",
            ".doc": "word",
            ".docx": "word",
            ".xls": "excel",
            ".xlsx": "excel",
            ".csv": "csv",
            ".ppt": "powerpoint",
            ".pptx": "powerpoint",
            ".md": "markdown",
            ".html": "html",
            ".htm": "html",
            ".txt": "text",
            ".rtf": "rtf",
            ".epub": "epub",
            ".eml": "email",
            ".msg": "email",
            ".png": "image",
            ".jpg": "image",
            ".jpeg": "image",
            ".gif": "image",
            ".bmp": "image",
        }
        return type_mapping.get(file_ext, "unknown")
    
    def _is_html_content(self, content: str) -> bool:
        """Check if content appears to be HTML."""
        html_indicators = ['<html', '<body', '<div', '<p>', '<table', '<!DOCTYPE']
        return any(indicator in content[:500] for indicator in html_indicators)
    
    async def _extract_pdf_tables(self, source_path: str) -> List[Dict[str, Any]]:
        """Extract tables from a PDF using pdfplumber with simple caching."""
        if source_path in self._table_cache:
            return self._table_cache[source_path]

        try:
            tables = await asyncio.to_thread(PDFTableExtractor.extract_tables, source_path)
        except Exception as exc:
            logger.warning(f"Failed to extract PDF tables for {source_path}: {exc}")
            tables = []

        self._table_cache[source_path] = tables
        return tables

    async def _extract_html_tables(self, source_path: str) -> List[Dict[str, Any]]:
        """Fetch tables from HTML sources using the Unstructured loader."""
        if not self.table_loader:
            return []

        if source_path in self._table_cache:
            return self._table_cache[source_path]

        try:
            tables = await asyncio.wait_for(
                self.table_loader.extract_tables(source_path),
                timeout=90,
            )
        except asyncio.TimeoutError:
            logger.warning(f"Table extraction timed out for {source_path}")
            tables = []
        except Exception as exc:
            logger.warning(f"Failed to extract tables for {source_path}: {exc}")
            tables = []

        self._table_cache[source_path] = tables
        return tables

    def _create_table_documents(
        self,
        base_metadata: Dict[str, Any],
        tables: List[Dict[str, Any]],
    ) -> List[Document]:
        """Convert table payloads into standalone Document instances."""
        table_docs: List[Document] = []
        sanitized_base = {
            key: value
            for key, value in base_metadata.items()
            if key not in {"tables"}
        }

        for idx, table in enumerate(tables):
            table_text = (table.get("markdown") or table.get("text") or "").strip()
            if not table_text:
                continue

            table_metadata = {
                **sanitized_base,
                "content_type": "table",
                "table_index": idx,
                "table_title": table.get("title"),
                "table_page": table.get("page_number"),
            }

            table_docs.append(
                Document(
                    page_content=table_text,
                    metadata=table_metadata,
                )
            )

        return table_docs


class DocumentLoaderFactory:
    """Factory for creating document loaders.
    
    This factory now uses LangChainDocumentLoader for most document types,
    falling back to custom loaders only when necessary.
    """
    
    def __init__(self):
        self.langchain_loader = LangChainDocumentLoader()
    
    async def create_loader(
        self,
        source: str,
        doc_type: DocumentType,
        **kwargs
    ) -> List[Document]:
        """Create appropriate loader based on document type."""
        try:
            if doc_type == DocumentType.WEB:
                # Use LangChainDocumentLoader for URL loading
                return await self.langchain_loader.load_from_url(source, extract_tables=kwargs.get('extract_tables', True))
                
            elif doc_type in [
                DocumentType.PDF,
                DocumentType.TEXT,
                DocumentType.MARKDOWN,
                DocumentType.DOCX,
                DocumentType.XLSX,
                DocumentType.CSV
            ]:
                # Use LangChainDocumentLoader for file loading
                return await self.langchain_loader.load_from_file(source)
                
            else:
                raise ValueError(f"Unsupported document type: {doc_type}")
                
        except Exception as e:
            logger.error(f"Failed to load document from {source}: {e}")
            raise


class CanadaCaScraper:
    """Specialized scraper for canada.ca travel instructions."""
    
    def __init__(self):
        """Initialize scraper."""
        self.base_url = settings.canada_ca_base_url
        self.travel_url = settings.travel_instructions_url
        self.loader_factory = DocumentLoaderFactory()
        
    async def scrape_travel_instructions(self) -> List[Document]:
        """Scrape all travel instruction pages."""
        documents = []
        
        try:
            # Load main page
            main_docs = await self.loader_factory.create_loader(
                self.travel_url,
                DocumentType.WEB
            )
            documents.extend(main_docs)
            
            # Extract links to sub-pages
            async with httpx.AsyncClient(timeout=settings.scraping_timeout) as client:
                response = await client.get(self.travel_url)
                soup = BeautifulSoup(response.text, "html.parser")
                
            # Find links to chapters/sections
            links = set()
            for link in soup.find_all("a", href=True):
                href = link["href"]
                
                # Filter for relevant links
                if "travel" in href.lower() or "cftdti" in href.lower():
                    if href.startswith("/"):
                        full_url = urljoin(self.base_url, href)
                    elif href.startswith("http"):
                        full_url = href
                    else:
                        full_url = urljoin(self.travel_url, href)
                        
                    # Only include canada.ca links
                    if "canada.ca" in full_url:
                        links.add(full_url)
                        
            logger.info(f"Found {len(links)} related pages to scrape")
            
            # Scrape each linked page
            for link in links:
                try:
                    link_docs = await self.loader_factory.create_loader(
                        link,
                        DocumentType.WEB
                    )
                    documents.extend(link_docs)
                    
                    # Be polite - add delay between requests
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Failed to scrape {link}: {e}")
                    continue
                    
            logger.info(f"Successfully scraped {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to scrape travel instructions: {e}")
            raise


class DocumentLoader:
    """Main document loader that routes to appropriate loaders.
    
    This class maintains backward compatibility while delegating to LangChainDocumentLoader.
    """
    
    def __init__(self):
        self.langchain_loader = LangChainDocumentLoader()
        self.loader_factory = DocumentLoaderFactory()
        
    async def load_from_url(self, url: str) -> List[Document]:
        """Load documents from a URL."""
        return await self.langchain_loader.load_from_url(url)
    
    async def load_from_file(self, file_path: str) -> List[Document]:
        """Load documents from a file."""
        return await self.langchain_loader.load_from_file(file_path)
    
    async def load(self, source: str, doc_type: DocumentType = None) -> List[Document]:
        """Load documents from any source."""
        if doc_type:
            return await self.loader_factory.create_loader(source, doc_type)
        
        # Auto-detect type
        if source.startswith(('http://', 'https://')):
            return await self.load_from_url(source)
        else:
            return await self.load_from_file(source)
