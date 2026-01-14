"""Parallel document ingestion optimizations."""

import asyncio
from typing import List, Tuple, Optional, Any, Dict
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from langchain_core.documents import Document as LangchainDocument
from langchain_core.embeddings import Embeddings

from app.core.logging import get_logger
from app.core.config import settings
from app.models.documents import Document
from app.services.progress_tracker import IngestionProgressTracker
from app.services.embedding_cache import EmbeddingCacheService

logger = get_logger(__name__)

# Try to import psutil for memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available - memory-aware batching disabled")


class MemoryMonitor:
    """Monitor system memory for adaptive resource allocation."""

    def __init__(self, sample_interval: int = 10):
        """Initialize memory monitor.

        Args:
            sample_interval: Number of operations between memory checks
        """
        self._operation_count = 0
        self._sample_interval = sample_interval
        self._last_available_mb = None
        self._enabled = PSUTIL_AVAILABLE and settings.memory_safe_mode

    def get_available_memory_mb(self) -> int:
        """Get available memory in MB.

        Uses cached value unless sample interval has passed.
        """
        if not self._enabled:
            return 2000  # Default assumption if psutil not available

        self._operation_count += 1

        # Only sample every N operations to reduce overhead
        if self._last_available_mb is None or self._operation_count >= self._sample_interval:
            self._operation_count = 0
            try:
                mem = psutil.virtual_memory()
                self._last_available_mb = int(mem.available / (1024 * 1024))
            except Exception as e:
                logger.warning(f"Failed to get memory info: {e}")
                self._last_available_mb = 2000  # Fallback

        return self._last_available_mb

    def is_memory_low(self) -> bool:
        """Check if available memory is below low threshold."""
        return self.get_available_memory_mb() < settings.memory_low_threshold_mb

    def is_memory_medium(self) -> bool:
        """Check if available memory is below medium threshold."""
        return self.get_available_memory_mb() < settings.memory_medium_threshold_mb

    def get_memory_factor(self) -> float:
        """Get a memory factor (0-1) for scaling operations.

        Returns:
            1.0 for plenty of memory, lower values as memory decreases
        """
        available = self.get_available_memory_mb()

        if available < settings.memory_low_threshold_mb:
            return 0.3
        elif available < settings.memory_medium_threshold_mb:
            return 0.6
        else:
            return 1.0


class AdaptiveConcurrencyManager:
    """Dynamically adjust concurrency based on workload characteristics."""

    def __init__(
        self,
        base_concurrency: int = 3,
        max_concurrency: int = 10,
        memory_monitor: Optional[MemoryMonitor] = None
    ):
        """Initialize adaptive concurrency manager.

        Args:
            base_concurrency: Default concurrency level
            max_concurrency: Maximum allowed concurrency
            memory_monitor: Memory monitor for memory-aware adjustments
        """
        self.base_concurrency = base_concurrency
        self.max_concurrency = max_concurrency
        self.memory_monitor = memory_monitor or MemoryMonitor()
        self._current_concurrency = base_concurrency

    def get_concurrency_for_batch(
        self,
        documents: List[Any],
        content_key: str = "content"
    ) -> int:
        """Calculate optimal concurrency for a batch of documents.

        Considers:
        1. Document sizes
        2. Document types (complex types need more resources)
        3. Available system memory

        Args:
            documents: List of documents to process
            content_key: Key to access document content

        Returns:
            Optimal concurrency level
        """
        if not documents:
            return self.base_concurrency

        # Factor 1: Average document size
        total_size = 0
        for doc in documents:
            content = doc.get(content_key, "") if isinstance(doc, dict) else getattr(doc, content_key, "")
            total_size += len(str(content))

        avg_size = total_size / len(documents)

        if avg_size < 5000:
            size_factor = 1.0
        elif avg_size < 20000:
            size_factor = 0.7
        else:
            size_factor = 0.4

        # Factor 2: Document type complexity
        complex_types = {"pdf", "docx", "xlsx"}
        complex_count = 0

        for doc in documents:
            doc_type = None
            if isinstance(doc, dict):
                metadata = doc.get("metadata", {})
                doc_type = metadata.get("type", "") if isinstance(metadata, dict) else ""
            elif hasattr(doc, "metadata"):
                metadata = doc.metadata
                if isinstance(metadata, dict):
                    doc_type = metadata.get("type", "")
                elif hasattr(metadata, "type"):
                    doc_type = metadata.type

            if doc_type and str(doc_type).lower() in complex_types:
                complex_count += 1

        complex_ratio = complex_count / len(documents)
        type_factor = 1.0 - (complex_ratio * 0.3)

        # Factor 3: Available memory
        memory_factor = self.memory_monitor.get_memory_factor()

        # Calculate optimal concurrency
        optimal = int(self.base_concurrency * size_factor * type_factor * memory_factor)

        # Apply bounds with hysteresis (don't change unless >20% different)
        if abs(optimal - self._current_concurrency) / max(1, self._current_concurrency) > 0.2:
            self._current_concurrency = max(1, min(optimal, self.max_concurrency))

        logger.debug(
            f"Adaptive concurrency: {self._current_concurrency} "
            f"(size_factor={size_factor:.2f}, type_factor={type_factor:.2f}, "
            f"memory_factor={memory_factor:.2f})"
        )

        return self._current_concurrency


# Global memory monitor instance
_memory_monitor = MemoryMonitor()


class ParallelEmbeddingGenerator:
    """Generate embeddings in parallel for faster processing."""
    
    def __init__(self, embeddings: Embeddings, max_workers: int = 8, cache_service: Optional[EmbeddingCacheService] = None):
        """Initialize parallel embedding generator."""
        self.embeddings = embeddings
        self.max_workers = max_workers
        self.executor = None  # Will be created dynamically
        self.cache_service = cache_service
        self._model_name = self._get_model_name()
        
    def _get_adaptive_workers(self, document_count: int) -> int:
        """Determine optimal number of workers based on document size."""
        if document_count < 10:
            return min(2, self.max_workers)
        elif document_count < 50:
            return min(4, self.max_workers)
        elif document_count < 100:
            return min(6, self.max_workers)
        elif document_count < 500:
            return min(8, self.max_workers)
        else:
            return min(12, self.max_workers)
            
    def _get_adaptive_batch_size(self, document_count: int) -> int:
        """Determine optimal batch size based on document count and available memory."""
        # Base batch size from document count
        if document_count < 10:
            base_size = 5
        elif document_count < 50:
            base_size = 10
        elif document_count < 100:
            base_size = 20
        elif document_count < 500:
            base_size = 30
        else:
            base_size = 50

        # Adjust based on available memory
        memory_factor = _memory_monitor.get_memory_factor()
        adjusted_size = max(5, int(base_size * memory_factor))

        if memory_factor < 1.0:
            logger.info(f"Memory-aware batch size: {adjusted_size} (factor: {memory_factor:.2f})")

        return adjusted_size
            
    def _ensure_executor(self, worker_count: int):
        """Ensure executor exists with appropriate worker count."""
        if self.executor is None or self.executor._max_workers != worker_count:
            if self.executor:
                self.executor.shutdown(wait=False)
            self.executor = ThreadPoolExecutor(max_workers=worker_count)
            logger.info(f"Created executor with {worker_count} workers")
        
    def _get_model_name(self) -> str:
        """Extract model name from embeddings instance."""
        if hasattr(self.embeddings, 'model'):
            return self.embeddings.model
        elif hasattr(self.embeddings, 'model_name'):
            return self.embeddings.model_name
        else:
            # Default fallback
            return "unknown"
        
    async def generate_embeddings_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        progress_callback: Optional[callable] = None
    ) -> List[List[float]]:
        """Generate embeddings in parallel batches with caching and adaptive processing."""
        try:
            # Determine adaptive parameters
            doc_count = len(texts)
            adaptive_workers = self._get_adaptive_workers(doc_count)
            adaptive_batch_size = batch_size or self._get_adaptive_batch_size(doc_count)
            
            # Ensure executor with appropriate worker count
            self._ensure_executor(adaptive_workers)
            
            logger.info(f"Processing {doc_count} texts with {adaptive_workers} workers and batch size {adaptive_batch_size}")
            # Check cache first if available
            if self.cache_service:
                cached_embeddings, missing_indices = await self.cache_service.get_embeddings_batch(
                    texts, self._model_name
                )
                
                # If all embeddings are cached, return them
                if not missing_indices:
                    logger.info(f"All {len(texts)} embeddings retrieved from cache")
                    if progress_callback:
                        await progress_callback(len(texts), len(texts))
                    return [cached_embeddings[i] for i in range(len(texts))]
                
                # Get texts that need embedding
                texts_to_embed = [texts[i] for i in missing_indices]
                logger.info(f"Found {len(cached_embeddings)} cached embeddings, need to generate {len(texts_to_embed)}")
            else:
                texts_to_embed = texts
                missing_indices = list(range(len(texts)))
                cached_embeddings = {}
            
            # Split texts into batches
            batches = [texts_to_embed[i:i + adaptive_batch_size] for i in range(0, len(texts_to_embed), adaptive_batch_size)]
            
            # Process batches in parallel
            loop = asyncio.get_event_loop()
            tasks = []
            
            for batch in batches:
                task = loop.run_in_executor(
                    self.executor,
                    self.embeddings.embed_documents,
                    batch
                )
                tasks.append(task)
            
            # Wait for all batches to complete
            batch_results = await asyncio.gather(*tasks)
            
            # Flatten results
            new_embeddings = []
            processed = len(cached_embeddings)
            for batch_result in batch_results:
                new_embeddings.extend(batch_result)
                processed += len(batch_result)
                if progress_callback:
                    await progress_callback(processed, len(texts))
            
            # Cache new embeddings
            if self.cache_service and new_embeddings:
                await self.cache_service.set_embeddings_batch(
                    texts_to_embed, self._model_name, new_embeddings
                )
            
            # Combine cached and new embeddings in correct order
            all_embeddings = []
            new_embedding_idx = 0
            for i in range(len(texts)):
                if i in cached_embeddings:
                    all_embeddings.append(cached_embeddings[i])
                else:
                    all_embeddings.append(new_embeddings[new_embedding_idx])
                    new_embedding_idx += 1
                
            logger.info(f"Generated {len(new_embeddings)} new embeddings, {len(cached_embeddings)} from cache")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings in parallel: {e}")
            raise
            
    async def generate_embeddings_concurrent(
        self,
        texts: List[str],
        max_concurrent: int = 20,
        progress_callback: Optional[callable] = None
    ) -> List[List[float]]:
        """Generate embeddings with controlled concurrency."""
        try:
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def embed_with_semaphore(text: str, index: int) -> Tuple[int, List[float]]:
                async with semaphore:
                    loop = asyncio.get_event_loop()
                    embedding = await loop.run_in_executor(
                        self.executor,
                        lambda: self.embeddings.embed_documents([text])[0]
                    )
                    return index, embedding
            
            # Create tasks for all texts
            tasks = [
                embed_with_semaphore(text, i) 
                for i, text in enumerate(texts)
            ]
            
            # Execute all tasks concurrently with progress tracking
            completed = 0
            embeddings_dict = {}
            
            for future in asyncio.as_completed(tasks):
                index, embedding = await future
                embeddings_dict[index] = embedding
                completed += 1
                
                if progress_callback:
                    await progress_callback(completed, len(texts))
            
            # Sort by index to maintain order
            embeddings = [embeddings_dict[i] for i in range(len(texts))]
            
            logger.info(f"Generated {len(embeddings)} embeddings with max concurrency {max_concurrent}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate concurrent embeddings: {e}")
            raise
            
    def close(self):
        """Shutdown the executor."""
        if self.executor:
            self.executor.shutdown(wait=True)
            logger.info("Parallel embedding generator executor shutdown")


class ParallelChunkProcessor:
    """Process document chunks in parallel with adaptive worker allocation."""
    
    def __init__(self, max_workers: int = 4):
        """Initialize parallel chunk processor."""
        self.max_workers = max_workers
        self.executor = None  # Will be created dynamically
        
    def _get_adaptive_workers(self, chunk_count: int) -> int:
        """Determine optimal number of workers based on chunk count."""
        if chunk_count < 10:
            return min(2, self.max_workers)
        elif chunk_count < 50:
            return min(3, self.max_workers)
        elif chunk_count < 100:
            return min(4, self.max_workers)
        else:
            return min(6, self.max_workers)
            
    def _ensure_executor(self, worker_count: int):
        """Ensure executor exists with appropriate worker count."""
        if self.executor is None or self.executor._max_workers != worker_count:
            if self.executor:
                self.executor.shutdown(wait=False)
            self.executor = ThreadPoolExecutor(max_workers=worker_count)
            logger.info(f"Chunk processor created executor with {worker_count} workers")
        
    async def process_chunks_parallel(
        self,
        chunks: List[LangchainDocument],
        process_func: callable,
        batch_size: int = 50
    ) -> List[Any]:
        """Process chunks in parallel batches with adaptive processing."""
        try:
            # Determine adaptive workers
            chunk_count = len(chunks)
            adaptive_workers = self._get_adaptive_workers(chunk_count)
            self._ensure_executor(adaptive_workers)
            
            logger.info(f"Processing {chunk_count} chunks with {adaptive_workers} workers")
            # Split chunks into batches
            batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
            
            # Process batches in parallel
            loop = asyncio.get_event_loop()
            tasks = []
            
            for batch in batches:
                task = loop.run_in_executor(
                    self.executor,
                    lambda b=batch: [process_func(chunk) for chunk in b]
                )
                tasks.append(task)
            
            # Wait for all batches to complete
            batch_results = await asyncio.gather(*tasks)
            
            # Flatten results
            results = []
            for batch_result in batch_results:
                results.extend(batch_result)
                
            logger.info(f"Processed {len(chunks)} chunks in {len(batches)} parallel batches")
            return results
            
        except Exception as e:
            logger.error(f"Failed to process chunks in parallel: {e}")
            raise
            
    async def enhance_chunks_metadata(
        self,
        chunks: List[LangchainDocument],
        enhancement_func: callable,
        max_concurrent: int = 10
    ) -> List[LangchainDocument]:
        """Enhance chunk metadata in parallel."""
        try:
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def enhance_with_semaphore(chunk: LangchainDocument, index: int) -> Tuple[int, LangchainDocument]:
                async with semaphore:
                    loop = asyncio.get_event_loop()
                    enhanced_chunk = await loop.run_in_executor(
                        self.executor,
                        enhancement_func,
                        chunk
                    )
                    return index, enhanced_chunk
            
            # Create tasks for all chunks
            tasks = [
                enhance_with_semaphore(chunk, i) 
                for i, chunk in enumerate(chunks)
            ]
            
            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks)
            
            # Sort by index to maintain order
            results.sort(key=lambda x: x[0])
            enhanced_chunks = [result[1] for result in results]
            
            logger.info(f"Enhanced {len(enhanced_chunks)} chunks with max concurrency {max_concurrent}")
            return enhanced_chunks
            
        except Exception as e:
            logger.error(f"Failed to enhance chunks in parallel: {e}")
            raise
            
    def close(self):
        """Shutdown the executor."""
        if self.executor:
            self.executor.shutdown(wait=True)
            logger.info("Chunk processor executor shutdown")


class OptimizedVectorStoreWriter:
    """Optimized vector store writer with parallel processing."""
    
    def __init__(self, vector_store, embeddings: Embeddings, progress_tracker: Optional[IngestionProgressTracker] = None, cache_service: Optional[EmbeddingCacheService] = None):
        """Initialize optimized writer."""
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.embedding_generator = ParallelEmbeddingGenerator(embeddings, cache_service=cache_service)
        self.progress_tracker = progress_tracker
        self.checkpoint_callback = None  # Will be set when needed
        
    def set_checkpoint_callback(self, callback: Optional[callable]):
        """Set callback for checkpoint updates."""
        self.checkpoint_callback = callback
        
    async def add_documents_optimized(
        self,
        documents: List[Document],
        batch_size: Optional[int] = None,
        embedding_batch_size: Optional[int] = None,
        max_concurrent_embeddings: Optional[int] = None
    ) -> List[str]:
        """Add documents with optimized parallel processing."""
        try:
            # Import settings here to avoid circular imports
            from app.core.config import settings
            
            # Use configured batch sizes or defaults
            batch_size = batch_size or settings.vector_store_batch_size
            embedding_batch_size = embedding_batch_size or settings.embedding_batch_size
            max_concurrent_embeddings = max_concurrent_embeddings or settings.max_concurrent_embeddings
            
            # Extract texts for embedding
            texts = [doc.content for doc in documents]
            
            # Generate embeddings in parallel
            logger.info(f"Generating embeddings for {len(texts)} documents...")
            
            # Update progress for embedding phase
            if self.progress_tracker:
                await self.progress_tracker.start_step("embedding")
            
            if len(texts) > 100:
                # Use batch processing for large document sets
                embeddings = await self.embedding_generator.generate_embeddings_batch(
                    texts, batch_size=embedding_batch_size,
                    progress_callback=self._update_embedding_progress if self.progress_tracker else None
                )
            else:
                # Use concurrent processing for smaller sets
                embeddings = await self.embedding_generator.generate_embeddings_concurrent(
                    texts, max_concurrent=max_concurrent_embeddings,
                    progress_callback=self._update_embedding_progress if self.progress_tracker else None
                )
                
            if self.progress_tracker:
                await self.progress_tracker.complete_step("embedding", f"Generated {len(embeddings)} embeddings")
            
            # Convert to numpy array for efficient storage
            embeddings_array = np.array(embeddings)
            
            # Prepare data for add_texts
            all_texts = []
            all_metadatas = []
            all_doc_ids = []
            
            for i, doc in enumerate(documents):
                metadata = doc.metadata.model_dump() if hasattr(doc.metadata, 'model_dump') else doc.metadata
                metadata["id"] = doc.id
                metadata["created_at"] = doc.created_at.isoformat()
                
                # Filter metadata
                filtered_metadata = {}
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        filtered_metadata[key] = value
                    elif isinstance(value, list) and key == "tags":
                        filtered_metadata[key] = ", ".join(str(v) for v in value)
                
                all_texts.append(doc.content)
                all_metadatas.append(filtered_metadata)
                all_doc_ids.append(doc.id)
            
            # Add to vector store with pre-computed embeddings
            logger.info(f"Adding {len(all_texts)} documents to vector store...")
            
            # Update progress for storing phase
            if self.progress_tracker:
                await self.progress_tracker.start_step("storing")
            
            all_ids = []
            loop = asyncio.get_event_loop()
            total_docs = len(all_texts)
            docs_stored = 0
            
            for i in range(0, len(all_texts), batch_size):
                batch_texts = all_texts[i:i + batch_size]
                batch_metadatas = all_metadatas[i:i + batch_size]
                batch_embeddings = embeddings_array[i:i + batch_size].tolist()
                batch_ids = all_doc_ids[i:i + batch_size]
                
                # Run synchronous add_texts in executor
                ids = await loop.run_in_executor(
                    None,
                    lambda: self.vector_store.add_texts(
                        texts=batch_texts,
                        metadatas=batch_metadatas,
                        embeddings=batch_embeddings,
                        ids=batch_ids
                    )
                )
                all_ids.extend(ids)
                docs_stored += len(batch_texts)
                logger.info(f"Added batch {i//batch_size + 1}: {len(batch_texts)} documents")
                
                # Update checkpoint with processed document IDs
                if self.checkpoint_callback:
                    for j, doc_id in enumerate(ids):
                        if doc_id:
                            await self.checkpoint_callback(doc_id)
                
                # Update storing progress
                if self.progress_tracker:
                    await self.progress_tracker.update_storing_progress(docs_stored, total_docs)
            
            return all_ids
            
        except Exception as e:
            logger.error(f"Failed to add documents with optimization: {e}")
            raise
            
    async def _update_embedding_progress(self, current: int, total: int):
        """Update embedding progress."""
        if self.progress_tracker:
            await self.progress_tracker.update_embedding_progress(current, total)
            
    def close(self):
        """Cleanup resources."""
        self.embedding_generator.close()