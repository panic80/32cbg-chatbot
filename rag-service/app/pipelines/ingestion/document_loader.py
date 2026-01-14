"""Document loading from various sources."""

from typing import List, Optional

from langchain_core.documents import Document as LangchainDocument

from app.core.logging import get_logger
from app.models.documents import DocumentIngestionRequest
from app.pipelines.loaders import LangChainDocumentLoader
from app.utils.retry import RetryManager, AGGRESSIVE_RETRY_CONFIG

logger = get_logger(__name__)


class DocumentLoader:
    """Loads documents from URLs, files, or direct content."""

    def __init__(self, retry_manager: Optional[RetryManager] = None):
        """Initialize document loader.

        Args:
            retry_manager: Optional retry manager for network operations.
        """
        self.retry_manager = retry_manager or RetryManager(AGGRESSIVE_RETRY_CONFIG)
        self._loader = LangChainDocumentLoader()

    async def load(
        self,
        request: DocumentIngestionRequest,
    ) -> List[LangchainDocument]:
        """Load documents from the request source.

        Args:
            request: The ingestion request with source info.

        Returns:
            List of loaded LangChain documents.

        Raises:
            Various exceptions on load failure.
        """
        return await self.retry_manager.execute_with_retry_async(
            lambda: self._load_documents(request)
        )

    async def _load_documents(
        self,
        request: DocumentIngestionRequest,
    ) -> List[LangchainDocument]:
        """Internal load implementation."""
        if request.content:
            # Direct content input
            return [
                LangchainDocument(
                    page_content=request.content,
                    metadata={
                        "source": "direct_input",
                        "type": request.type,
                        **(request.metadata or {}),
                    },
                )
            ]

        if request.url:
            logger.info(f"Loading document from URL: {request.url}")
            return await self._loader.load_from_url(request.url)

        if request.file_path:
            logger.info(f"Loading document from file: {request.file_path}")
            return await self._loader.load_from_file(request.file_path)

        raise ValueError("Request must have content, url, or file_path")

    async def load_from_url(self, url: str) -> List[LangchainDocument]:
        """Load documents from a URL.

        Args:
            url: The URL to load from.

        Returns:
            List of loaded documents.
        """
        return await self.retry_manager.execute_with_retry_async(
            lambda: self._loader.load_from_url(url)
        )

    async def load_from_file(self, file_path: str) -> List[LangchainDocument]:
        """Load documents from a file.

        Args:
            file_path: Path to the file.

        Returns:
            List of loaded documents.
        """
        return await self._loader.load_from_file(file_path)

    def load_from_content(
        self,
        content: str,
        metadata: Optional[dict] = None,
    ) -> List[LangchainDocument]:
        """Load document from direct content.

        Args:
            content: The document content.
            metadata: Optional metadata to attach.

        Returns:
            List with single document.
        """
        return [
            LangchainDocument(
                page_content=content,
                metadata=metadata or {"source": "direct_input"},
            )
        ]
