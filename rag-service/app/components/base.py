"""Base classes for RAG components following LangChain patterns."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Callable
import asyncio
from datetime import datetime, timezone

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.language_models import BaseLLM
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field

from app.core.logging import get_logger
from app.utils.langchain_utils import langchain_retry

logger = get_logger(__name__)


class BaseComponent:
    """
    Base class for all RAG components.
    
    Provides common functionality like logging, metrics, and component identification.
    """
    
    def __init__(
        self,
        component_type: str = "base",
        component_name: str = "unnamed",
        **kwargs
    ):
        """
        Initialize the base component.
        
        Args:
            component_type: Type of component (e.g., 'retriever', 'reranker', 'loader')
            component_name: Name of the component instance
            **kwargs: Additional keyword arguments
        """
        self.component_type = component_type
        self.component_name = component_name
        self._events = []
        self._metrics = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "events_logged": 0,
            "errors": 0
        }
        
        # Log component initialization
        logger.info(
            f"Initialized {component_type} component: {component_name}"
        )
    
    def _log_event(
        self,
        event_type: str,
        event_data: Dict[str, Any],
        level: str = "info"
    ) -> None:
        """
        Log an event for this component.
        
        Args:
            event_type: Type of event (e.g., 'query', 'retrieve', 'error')
            event_data: Data associated with the event
            level: Log level ('debug', 'info', 'warning', 'error')
        """
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "component_type": self.component_type,
            "component_name": self.component_name,
            "event_type": event_type,
            "data": event_data
        }
        
        self._events.append(event)
        self._metrics["events_logged"] += 1
        
        # Log to logger
        log_message = (
            f"[{self.component_type}:{self.component_name}] "
            f"{event_type}: {event_data}"
        )
        
        if level == "debug":
            logger.debug(log_message)
        elif level == "warning":
            logger.warning(log_message)
        elif level == "error":
            logger.error(log_message)
            self._metrics["errors"] += 1
        else:
            logger.info(log_message)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get metrics for this component.
        
        Returns:
            Dictionary of component metrics
        """
        return {
            "component_type": self.component_type,
            "component_name": self.component_name,
            "metrics": self._metrics.copy(),
            "recent_events": self._events[-10:]  # Last 10 events
        }
    
    def reset_metrics(self) -> None:
        """Reset component metrics."""
        self._metrics = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "events_logged": 0,
            "errors": 0
        }
        self._events = []
        logger.info(
            f"Reset metrics for {self.component_type}:{self.component_name}"
        )
    
    @staticmethod
    def monitor_performance(func: Callable) -> Callable:
        """
        Decorator to monitor performance of component methods.
        
        Args:
            func: Function to monitor
            
        Returns:
            Wrapped function with performance monitoring
        """
        import time
        import functools
        
        @functools.wraps(func)
        def sync_wrapper(self, *args, **kwargs):
            start_time = time.time()
            try:
                result = func(self, *args, **kwargs)
                elapsed = time.time() - start_time
                if hasattr(self, '_log_event'):
                    self._log_event(
                        "performance",
                        {
                            "method": func.__name__,
                            "duration_ms": elapsed * 1000,
                            "status": "success"
                        },
                        level="debug"
                    )
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                if hasattr(self, '_log_event'):
                    self._log_event(
                        "performance",
                        {
                            "method": func.__name__,
                            "duration_ms": elapsed * 1000,
                            "status": "error",
                            "error": str(e)
                        },
                        level="error"
                    )
                raise
        
        @functools.wraps(func)
        async def async_wrapper(self, *args, **kwargs):
            start_time = time.time()
            try:
                result = await func(self, *args, **kwargs)
                elapsed = time.time() - start_time
                if hasattr(self, '_log_event'):
                    self._log_event(
                        "performance",
                        {
                            "method": func.__name__,
                            "duration_ms": elapsed * 1000,
                            "status": "success"
                        },
                        level="debug"
                    )
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                if hasattr(self, '_log_event'):
                    self._log_event(
                        "performance",
                        {
                            "method": func.__name__,
                            "duration_ms": elapsed * 1000,
                            "status": "error",
                            "error": str(e)
                        },
                        level="error"
                    )
                raise
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper


class BaseRAGRetriever(BaseRetriever, ABC):
    """
    Base retriever class for RAG components.
    
    Extends LangChain's BaseRetriever with additional functionality
    for metrics, caching, and error handling.
    """
    
    # Configuration
    name: str = Field(default="BaseRAGRetriever", description="Retriever name")
    description: str = Field(default="Base RAG retriever", description="Retriever description")
    
    # Metrics
    _call_count: int = 0
    _total_latency: float = 0.0
    _error_count: int = 0
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
        **kwargs: Any
    ) -> List[Document]:
        """Synchronous retrieval method."""
        # Run async version in event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self._aget_relevant_documents(query, run_manager=run_manager, **kwargs)
        )
    
    @abstractmethod
    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
        **kwargs: Any
    ) -> List[Document]:
        """
        Async retrieval method to be implemented by subclasses.
        
        Args:
            query: The search query
            run_manager: Callback manager for the retriever run
            **kwargs: Additional keyword arguments
            
        Returns:
            List of relevant documents
        """
        pass
    
    @langchain_retry(max_attempts=3, initial_wait=1.0)
    async def aget_relevant_documents_with_retry(
        self,
        query: str,
        **kwargs: Any
    ) -> List[Document]:
        """Get relevant documents with retry logic."""
        start_time = datetime.now(timezone.utc)
        
        try:
            self._call_count += 1
            docs = await self.aget_relevant_documents(query, **kwargs)
            
            # Update metrics
            latency = (datetime.now(timezone.utc) - start_time).total_seconds()
            self._total_latency += latency
            
            logger.info(
                f"{self.name} retrieved {len(docs)} documents in {latency:.2f}s "
                f"for query: {query[:50]}..."
            )
            
            return docs
            
        except Exception as e:
            self._error_count += 1
            logger.error(f"{self.name} retrieval failed: {e}")
            raise
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get retriever metrics."""
        avg_latency = (
            self._total_latency / self._call_count 
            if self._call_count > 0 else 0
        )
        
        return {
            "name": self.name,
            "call_count": self._call_count,
            "error_count": self._error_count,
            "error_rate": self._error_count / max(self._call_count, 1),
            "avg_latency": avg_latency,
            "total_latency": self._total_latency
        }
    
    def reset_metrics(self) -> None:
        """Reset metrics."""
        self._call_count = 0
        self._total_latency = 0.0
        self._error_count = 0


class BaseRAGLoader(BaseLoader, ABC):
    """
    Base loader class for RAG components.
    
    Extends LangChain's BaseLoader with additional functionality
    for preprocessing, validation, and metadata enrichment.
    """
    
    # Configuration
    name: str = "BaseRAGLoader"
    description: str = "Base RAG document loader"
    
    # Processing options
    preprocess: bool = True
    validate: bool = True
    enrich_metadata: bool = True
    
    # Preprocessing functions
    preprocessors: List[Callable[[str], str]] = []
    
    def __init__(
        self,
        preprocess: bool = True,
        validate: bool = True,
        enrich_metadata: bool = True,
        **kwargs
    ):
        """Initialize the loader."""
        super().__init__(**kwargs)
        self.preprocess = preprocess
        self.validate = validate
        self.enrich_metadata = enrich_metadata
    
    @abstractmethod
    def load(self) -> List[Document]:
        """Load documents."""
        pass
    
    async def aload(self) -> List[Document]:
        """Async load documents."""
        # Default implementation runs sync version in thread
        return await asyncio.to_thread(self.load)
    
    def lazy_load(self) -> List[Document]:
        """Lazy load documents (default to regular load)."""
        return self.load()
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text using configured preprocessors."""
        if not self.preprocess:
            return text
            
        for preprocessor in self.preprocessors:
            try:
                text = preprocessor(text)
            except Exception as e:
                logger.warning(f"Preprocessor {preprocessor.__name__} failed: {e}")
        
        return text
    
    def _validate_document(self, doc: Document) -> bool:
        """Validate a document."""
        if not self.validate:
            return True
            
        # Basic validation
        if not doc.page_content or not doc.page_content.strip():
            return False
            
        # Check minimum length
        if len(doc.page_content) < 10:
            return False
            
        return True
    
    def _enrich_metadata(self, doc: Document) -> Document:
        """Enrich document metadata."""
        if not self.enrich_metadata:
            return doc
            
        # Add common metadata
        if "loaded_at" not in doc.metadata:
            doc.metadata["loaded_at"] = datetime.now(timezone.utc).isoformat()
        
        if "loader" not in doc.metadata:
            doc.metadata["loader"] = self.name
            
        if "char_count" not in doc.metadata:
            doc.metadata["char_count"] = len(doc.page_content)
            
        if "word_count" not in doc.metadata:
            doc.metadata["word_count"] = len(doc.page_content.split())
        
        return doc
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """Process documents with preprocessing, validation, and enrichment."""
        processed_docs = []
        
        for doc in documents:
            try:
                # Preprocess
                doc.page_content = self._preprocess_text(doc.page_content)
                
                # Validate
                if not self._validate_document(doc):
                    logger.debug(f"Document failed validation: {doc.metadata}")
                    continue
                
                # Enrich
                doc = self._enrich_metadata(doc)
                
                processed_docs.append(doc)
                
            except Exception as e:
                logger.error(f"Error processing document: {e}")
                continue
        
        logger.info(
            f"{self.name} processed {len(processed_docs)}/{len(documents)} documents"
        )
        
        return processed_docs


class BaseChunker(ABC):
    """Base class for document chunkers."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        **kwargs
    ):
        """Initialize the chunker."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    @abstractmethod
    def chunk_documents(
        self,
        documents: List[Document]
    ) -> List[Document]:
        """Chunk documents into smaller pieces."""
        pass
    
    async def achunk_documents(
        self,
        documents: List[Document]
    ) -> List[Document]:
        """Async chunk documents."""
        return await asyncio.to_thread(self.chunk_documents, documents)
    
    def _copy_metadata(
        self,
        source_doc: Document,
        chunk_index: int,
        total_chunks: int
    ) -> Dict[str, Any]:
        """Copy and enrich metadata for a chunk."""
        metadata = source_doc.metadata.copy()
        metadata.update({
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "original_id": source_doc.metadata.get("id", "unknown"),
            "chunking_method": self.__class__.__name__
        })
        return metadata


class BaseEmbedder(ABC):
    """Base class for document embedders."""
    
    def __init__(
        self,
        embeddings: Embeddings,
        batch_size: int = 100,
        **kwargs
    ):
        """Initialize the embedder."""
        self.embeddings = embeddings
        self.batch_size = batch_size
    
    @abstractmethod
    async def embed_documents(
        self,
        documents: List[Document]
    ) -> List[Tuple[Document, List[float]]]:
        """Embed documents and return tuples of (document, embedding)."""
        pass
    
    def embed_documents_sync(
        self,
        documents: List[Document]
    ) -> List[Tuple[Document, List[float]]]:
        """Synchronous version of embed_documents."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.embed_documents(documents))


class BasePipeline(ABC):
    """Base class for processing pipelines."""
    
    def __init__(
        self,
        name: str = "BasePipeline",
        **kwargs
    ):
        """Initialize the pipeline."""
        self.name = name
        self.components: Dict[str, Any] = {}
    
    @abstractmethod
    async def run(
        self,
        input_data: Any,
        **kwargs
    ) -> Any:
        """Run the pipeline."""
        pass
    
    def run_sync(self, input_data: Any, **kwargs) -> Any:
        """Synchronous version of run."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.run(input_data, **kwargs))
    
    def add_component(self, name: str, component: Any) -> None:
        """Add a component to the pipeline."""
        self.components[name] = component
        logger.info(f"Added component '{name}' to {self.name}")
    
    def get_component(self, name: str) -> Any:
        """Get a component by name."""
        return self.components.get(name)
    
    def list_components(self) -> List[str]:
        """List all component names."""
        return list(self.components.keys())
