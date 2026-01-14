"""Document chunking/splitting strategies."""

import asyncio
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor

from langchain_core.documents import Document as LangchainDocument

from app.core.errors import ParsingError
from app.core.logging import get_logger
from app.models.documents import DocumentType
from app.pipelines.splitters import LangChainTextSplitter
from app.pipelines.smart_splitters import SmartDocumentSplitter

logger = get_logger(__name__)


class DocumentChunker:
    """Splits documents into chunks using various strategies."""

    def __init__(
        self,
        use_smart_chunking: bool = True,
        max_workers: int = 4,
    ):
        """Initialize chunker.

        Args:
            use_smart_chunking: Whether to use smart document-aware chunking.
            max_workers: Max parallel workers for chunk processing.
        """
        self.use_smart_chunking = use_smart_chunking
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def close(self):
        """Clean up resources."""
        self._executor.shutdown(wait=False)

    async def chunk_documents(
        self,
        documents: List[LangchainDocument],
        progress_callback: Optional[callable] = None,
    ) -> List[LangchainDocument]:
        """Split documents into chunks.

        Args:
            documents: Documents to split.
            progress_callback: Optional callback for progress updates.

        Returns:
            List of document chunks.

        Raises:
            ParsingError: If splitting fails.
        """
        try:
            splitter = self._get_splitter()

            # For large documents, split in parallel
            total_size = sum(len(doc.page_content) for doc in documents)
            if len(documents) > 5 or total_size > 50000:
                return await self._chunk_parallel(
                    documents, splitter, progress_callback
                )
            else:
                return await self._chunk_sequential(documents, splitter)

        except Exception as e:
            raise ParsingError(
                f"Failed to split documents: {e}",
                document_type="multiple",
            )

    def _get_splitter(self):
        """Get the appropriate splitter based on configuration."""
        if self.use_smart_chunking:
            return SmartDocumentSplitter()
        return LangChainTextSplitter()

    async def _chunk_sequential(
        self,
        documents: List[LangchainDocument],
        splitter,
    ) -> List[LangchainDocument]:
        """Chunk documents sequentially."""
        if not documents:
            return []

        doc_type = documents[0].metadata.get("type", DocumentType.TEXT)

        if isinstance(splitter, SmartDocumentSplitter):
            chunks = []
            for doc in documents:
                doc_chunks = splitter.split_by_type(doc, doc_type)
                chunks.extend(doc_chunks)
            return chunks
        else:
            return splitter.split_documents(documents)

    async def _chunk_parallel(
        self,
        documents: List[LangchainDocument],
        splitter,
        progress_callback: Optional[callable] = None,
    ) -> List[LangchainDocument]:
        """Chunk documents in parallel."""
        logger.info(f"Using parallel processing to split {len(documents)} documents")

        def split_single_doc(doc: LangchainDocument) -> List[LangchainDocument]:
            doc_type = doc.metadata.get("type", DocumentType.TEXT)
            if isinstance(splitter, SmartDocumentSplitter):
                return splitter.split_by_type(doc, doc_type)
            else:
                return splitter.split_documents([doc])

        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(self._executor, split_single_doc, doc)
            for doc in documents
        ]

        completed = 0
        chunks = []
        for future in asyncio.as_completed(tasks):
            doc_chunks = await future
            chunks.extend(doc_chunks)
            completed += 1

            if progress_callback:
                await progress_callback(completed, len(documents))

        logger.info(f"Parallel splitting produced {len(chunks)} chunks")
        return chunks

    def chunk_single(
        self,
        document: LangchainDocument,
        doc_type: Optional[DocumentType] = None,
    ) -> List[LangchainDocument]:
        """Chunk a single document synchronously.

        Args:
            document: The document to chunk.
            doc_type: Optional document type hint.

        Returns:
            List of chunks.
        """
        splitter = self._get_splitter()
        effective_type = doc_type or document.metadata.get("type", DocumentType.TEXT)

        if isinstance(splitter, SmartDocumentSplitter):
            return splitter.split_by_type(document, effective_type)
        else:
            return splitter.split_documents([document])
