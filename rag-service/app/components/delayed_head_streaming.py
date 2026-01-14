"""
Delayed Head Streaming Handler for intelligent document streaming.

This component optimizes user experience by streaming high-confidence documents
immediately while continuing background retrieval and processing for remaining results.

Streaming strategy:
- Stream top-ranked documents immediately when confidence is high
- Continue background processing for remaining documents
- Handle streaming interruption gracefully
- Provide real-time streaming status and completion metrics
"""

import asyncio
import time
from typing import List, Dict, Optional, Tuple, AsyncIterator, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging

from langchain_core.documents import Document

from app.components.rrf_merger import RRFDocument
from app.components.uncertainty_scorer import UncertaintyResult
from app.core.logging import get_logger

logger = get_logger(__name__)


class StreamingDecision(Enum):
    """Streaming strategy decisions."""
    IMMEDIATE = "immediate"     # Stream all results immediately
    DELAYED = "delayed"         # Stream head, continue processing
    BATCH = "batch"            # Wait for all results before streaming
    PROGRESSIVE = "progressive" # Stream incrementally as ready


@dataclass
class StreamingMetrics:
    """Metrics from streaming operation."""
    total_documents_streamed: int = 0
    head_documents_count: int = 0
    background_documents_count: int = 0
    
    # Timing metrics
    first_document_latency_ms: float = 0.0
    head_completion_time_ms: float = 0.0
    total_completion_time_ms: float = 0.0
    
    # Quality metrics
    head_confidence_score: float = 0.0
    streaming_efficiency_ratio: float = 0.0  # head_time / total_time
    
    # Background processing
    background_processing_time_ms: float = 0.0
    background_success: bool = True
    background_error: Optional[str] = None


@dataclass
class StreamingConfiguration:
    """Configuration for delayed streaming."""
    enable_delayed_streaming: bool = True
    head_document_count: int = 3           # Number of documents to stream immediately
    confidence_threshold: float = 0.7      # Minimum confidence for delayed streaming
    max_head_latency_ms: float = 500.0     # Maximum time to wait for head documents
    background_timeout_ms: float = 10000.0 # Maximum background processing time
    min_score_gap: float = 0.1             # Minimum score gap to justify head streaming
    enable_progressive_streaming: bool = False  # Stream documents as they become available


@dataclass
class StreamingChunk:
    """A chunk of documents to stream."""
    documents: List[Document]
    chunk_type: str  # 'head', 'background', 'final'
    confidence_score: float
    is_complete: bool = False
    metadata: Optional[Dict[str, Any]] = None


class DelayedHeadStreamingHandler:
    """
    Handles intelligent streaming of retrieval results.
    
    This component analyzes retrieval results to determine the optimal streaming
    strategy, balancing user experience (quick initial results) with completeness.
    """
    
    def __init__(
        self,
        config: Optional[StreamingConfiguration] = None,
        background_processor: Optional[Callable] = None
    ):
        """
        Initialize delayed head streaming handler.
        
        Args:
            config: Streaming configuration parameters
            background_processor: Optional async function for background processing
        """
        self.config = config or StreamingConfiguration()
        self.background_processor = background_processor
        
        # Streaming state
        self._active_streams = {}  # stream_id -> streaming_state
        self._stream_counter = 0
        
        # Statistics
        self.total_streams = 0
        self.successful_streams = 0
        self.average_head_latency_ms = 0.0
        
        logger.info(f"DelayedHeadStreamingHandler initialized: head_count={self.config.head_document_count}, "
                   f"confidence_threshold={self.config.confidence_threshold}")
    
    async def stream_documents(
        self,
        documents: List[RRFDocument],
        query: str,
        uncertainty_result: Optional[UncertaintyResult] = None,
        stream_id: Optional[str] = None
    ) -> AsyncIterator[StreamingChunk]:
        """
        Stream documents with intelligent head/background splitting.
        
        Args:
            documents: RRF-scored documents to stream
            query: Original query for context
            uncertainty_result: Query uncertainty analysis
            stream_id: Optional stream identifier for tracking
            
        Yields:
            StreamingChunk: Chunks of documents to stream
        """
        if not stream_id:
            stream_id = f"stream_{self._stream_counter}"
            self._stream_counter += 1
        
        start_time = time.time()
        metrics = StreamingMetrics()
        
        try:
            # Phase 1: Analyze streaming decision
            decision = self._make_streaming_decision(documents, query, uncertainty_result)
            logger.debug(f"Streaming decision for {stream_id}: {decision.value}")
            
            # Phase 2: Execute streaming strategy
            if decision == StreamingDecision.IMMEDIATE:
                async for chunk in self._stream_immediate(documents, metrics):
                    yield chunk
            elif decision == StreamingDecision.DELAYED:
                async for chunk in self._stream_delayed(documents, query, metrics):
                    yield chunk
            elif decision == StreamingDecision.PROGRESSIVE:
                async for chunk in self._stream_progressive(documents, metrics):
                    yield chunk
            else:  # BATCH
                async for chunk in self._stream_batch(documents, metrics):
                    yield chunk
            
            # Phase 3: Finalize metrics
            total_time = (time.time() - start_time) * 1000
            metrics.total_completion_time_ms = total_time
            
            if metrics.head_completion_time_ms > 0:
                metrics.streaming_efficiency_ratio = metrics.head_completion_time_ms / total_time
            
            self._update_statistics(metrics)
            self.successful_streams += 1
            
            logger.info(f"Streaming completed for {stream_id}: {metrics.total_documents_streamed} docs "
                       f"in {total_time:.1f}ms (head: {metrics.head_documents_count})")
            
        except Exception as e:
            logger.error(f"Streaming error for {stream_id}: {e}")
            # Yield error chunk
            yield StreamingChunk(
                documents=[],
                chunk_type="error",
                confidence_score=0.0,
                is_complete=True,
                metadata={"error": str(e)}
            )
        finally:
            self.total_streams += 1
            if stream_id in self._active_streams:
                del self._active_streams[stream_id]
    
    def _make_streaming_decision(
        self,
        documents: List[RRFDocument],
        query: str,
        uncertainty_result: Optional[UncertaintyResult] = None
    ) -> StreamingDecision:
        """Make intelligent decision about streaming strategy."""
        
        if not self.config.enable_delayed_streaming:
            return StreamingDecision.IMMEDIATE
        
        if len(documents) < self.config.head_document_count:
            return StreamingDecision.IMMEDIATE
        
        # Analyze document quality and score distribution
        if len(documents) == 0:
            return StreamingDecision.BATCH
        
        # Calculate confidence in head documents
        head_confidence = self._calculate_head_confidence(documents)
        
        if head_confidence < self.config.confidence_threshold:
            return StreamingDecision.BATCH
        
        # Check score separation
        head_docs = documents[:self.config.head_document_count]
        remaining_docs = documents[self.config.head_document_count:]
        
        if remaining_docs:
            score_gap = head_docs[-1].rrf_score - remaining_docs[0].rrf_score
            if score_gap < self.config.min_score_gap:
                return StreamingDecision.BATCH
        
        # Check query complexity
        uncertainty_score = uncertainty_result.overall_uncertainty if uncertainty_result else 0.5
        
        if uncertainty_score >= 0.8:  # High uncertainty queries benefit from complete results
            return StreamingDecision.BATCH
        
        # Progressive streaming for medium confidence
        if self.config.enable_progressive_streaming and head_confidence < 0.9:
            return StreamingDecision.PROGRESSIVE
        
        return StreamingDecision.DELAYED
    
    def _calculate_head_confidence(self, documents: List[RRFDocument]) -> float:
        """Calculate confidence in head documents."""
        if not documents:
            return 0.0
        
        head_docs = documents[:self.config.head_document_count]
        
        # Base confidence on RRF scores
        avg_score = sum(doc.rrf_score for doc in head_docs) / len(head_docs)
        
        # Bonus for high top score
        top_score_bonus = min(documents[0].rrf_score, 0.2)
        
        # Bonus for score consistency in head
        if len(head_docs) > 1:
            score_variance = sum((doc.rrf_score - avg_score) ** 2 for doc in head_docs) / len(head_docs)
            consistency_bonus = max(0, 0.1 - score_variance)
        else:
            consistency_bonus = 0.0
        
        total_confidence = min(avg_score + top_score_bonus + consistency_bonus, 1.0)
        return total_confidence
    
    async def _stream_immediate(
        self,
        documents: List[RRFDocument],
        metrics: StreamingMetrics
    ) -> AsyncIterator[StreamingChunk]:
        """Stream all documents immediately."""
        start_time = time.time()
        
        doc_list = [Document(page_content=doc.document.page_content, metadata=doc.document.metadata)
                   for doc in documents]
        
        # Single chunk with all documents
        chunk = StreamingChunk(
            documents=doc_list,
            chunk_type="immediate",
            confidence_score=1.0,
            is_complete=True,
            metadata={"strategy": "immediate"}
        )
        
        first_doc_time = (time.time() - start_time) * 1000
        metrics.first_document_latency_ms = first_doc_time
        metrics.total_documents_streamed = len(doc_list)
        metrics.head_documents_count = len(doc_list)
        
        yield chunk
    
    async def _stream_delayed(
        self,
        documents: List[RRFDocument],
        query: str,
        metrics: StreamingMetrics
    ) -> AsyncIterator[StreamingChunk]:
        """Stream head documents immediately, process remainder in background."""
        start_time = time.time()
        
        # Split documents
        head_docs = documents[:self.config.head_document_count]
        background_docs = documents[self.config.head_document_count:]
        
        # Stream head documents immediately
        head_doc_list = [Document(page_content=doc.document.page_content, metadata=doc.document.metadata)
                        for doc in head_docs]
        
        head_chunk = StreamingChunk(
            documents=head_doc_list,
            chunk_type="head",
            confidence_score=self._calculate_head_confidence(documents),
            is_complete=False,
            metadata={"strategy": "delayed", "total_docs": len(documents)}
        )
        
        # Update metrics for head
        head_time = (time.time() - start_time) * 1000
        metrics.first_document_latency_ms = head_time
        metrics.head_completion_time_ms = head_time
        metrics.head_documents_count = len(head_doc_list)
        metrics.head_confidence_score = head_chunk.confidence_score
        
        yield head_chunk
        
        # Process background documents
        if background_docs:
            background_start = time.time()
            
            try:
                # Process background documents (could involve additional retrieval/reranking)
                processed_background = await self._process_background_documents(
                    background_docs, query
                )
                
                background_doc_list = [
                    Document(page_content=doc.document.page_content, metadata=doc.document.metadata)
                    for doc in processed_background
                ]
                
                background_chunk = StreamingChunk(
                    documents=background_doc_list,
                    chunk_type="background",
                    confidence_score=0.6,  # Lower confidence for background
                    is_complete=True,
                    metadata={"strategy": "delayed"}
                )
                
                # Update metrics for background
                background_time = (time.time() - background_start) * 1000
                metrics.background_processing_time_ms = background_time
                metrics.background_documents_count = len(background_doc_list)
                metrics.background_success = True
                
                yield background_chunk
                
            except Exception as e:
                logger.warning(f"Background processing failed: {e}")
                metrics.background_success = False
                metrics.background_error = str(e)
                
                # Still yield remaining documents as-is
                background_doc_list = [
                    Document(page_content=doc.document.page_content, metadata=doc.document.metadata)
                    for doc in background_docs
                ]
                
                fallback_chunk = StreamingChunk(
                    documents=background_doc_list,
                    chunk_type="background",
                    confidence_score=0.4,  # Lower confidence due to processing failure
                    is_complete=True,
                    metadata={"strategy": "delayed", "background_error": str(e)}
                )
                
                yield fallback_chunk
        
        # Update total metrics
        metrics.total_documents_streamed = len(documents)
    
    async def _stream_progressive(
        self,
        documents: List[RRFDocument],
        metrics: StreamingMetrics
    ) -> AsyncIterator[StreamingChunk]:
        """Stream documents progressively as they become available."""
        start_time = time.time()
        
        # Simulate progressive availability (in real implementation, 
        # this would be driven by actual retrieval completion)
        batch_size = max(1, len(documents) // 3)
        
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            doc_list = [Document(page_content=doc.document.page_content, metadata=doc.document.metadata)
                       for doc in batch_docs]
            
            is_complete = (i + batch_size) >= len(documents)
            
            chunk = StreamingChunk(
                documents=doc_list,
                chunk_type="progressive",
                confidence_score=0.8 - (i / len(documents)) * 0.3,  # Decreasing confidence
                is_complete=is_complete,
                metadata={"strategy": "progressive", "batch_index": i // batch_size}
            )
            
            if i == 0:  # First batch
                first_doc_time = (time.time() - start_time) * 1000
                metrics.first_document_latency_ms = first_doc_time
            
            yield chunk
            
            # Small delay between progressive chunks
            if not is_complete:
                await asyncio.sleep(0.05)
        
        metrics.total_documents_streamed = len(documents)
    
    async def _stream_batch(
        self,
        documents: List[RRFDocument],
        metrics: StreamingMetrics
    ) -> AsyncIterator[StreamingChunk]:
        """Wait for all processing to complete before streaming."""
        start_time = time.time()
        
        # Simulate batch processing delay
        await asyncio.sleep(0.1)
        
        doc_list = [Document(page_content=doc.document.page_content, metadata=doc.document.metadata)
                   for doc in documents]
        
        chunk = StreamingChunk(
            documents=doc_list,
            chunk_type="batch",
            confidence_score=1.0,  # High confidence since we waited for complete processing
            is_complete=True,
            metadata={"strategy": "batch"}
        )
        
        batch_time = (time.time() - start_time) * 1000
        metrics.first_document_latency_ms = batch_time
        metrics.total_documents_streamed = len(doc_list)
        
        yield chunk
    
    async def _process_background_documents(
        self,
        background_docs: List[RRFDocument],
        query: str
    ) -> List[RRFDocument]:
        """Process background documents (placeholder for additional processing)."""
        
        # If background processor is provided, use it
        if self.background_processor:
            try:
                processed = await asyncio.wait_for(
                    self.background_processor(query, background_docs),
                    timeout=self.config.background_timeout_ms / 1000.0
                )
                return processed
            except asyncio.TimeoutError:
                logger.warning("Background processor timed out")
                return background_docs
            except Exception as e:
                logger.warning(f"Background processor failed: {e}")
                return background_docs
        
        # Default: return documents as-is
        return background_docs
    
    def _update_statistics(self, metrics: StreamingMetrics):
        """Update running statistics."""
        if self.total_streams == 0:
            self.average_head_latency_ms = metrics.first_document_latency_ms
        else:
            # Running average
            alpha = 1.0 / (self.total_streams + 1)
            self.average_head_latency_ms = (
                (1 - alpha) * self.average_head_latency_ms +
                alpha * metrics.first_document_latency_ms
            )
    
    def get_streaming_statistics(self) -> Dict[str, Any]:
        """Get current streaming statistics."""
        success_rate = self.successful_streams / max(1, self.total_streams)
        
        return {
            "total_streams": self.total_streams,
            "successful_streams": self.successful_streams,
            "success_rate": success_rate,
            "average_head_latency_ms": self.average_head_latency_ms,
            "active_streams": len(self._active_streams),
            "configuration": {
                "head_document_count": self.config.head_document_count,
                "confidence_threshold": self.config.confidence_threshold,
                "max_head_latency_ms": self.config.max_head_latency_ms,
                "enable_progressive": self.config.enable_progressive_streaming
            }
        }
    
    def reset_statistics(self):
        """Reset streaming statistics."""
        self.total_streams = 0
        self.successful_streams = 0
        self.average_head_latency_ms = 0.0


def create_delayed_head_streaming_handler(
    config: Optional[StreamingConfiguration] = None,
    **kwargs
) -> DelayedHeadStreamingHandler:
    """
    Factory function to create a delayed head streaming handler.
    
    Args:
        config: Streaming configuration
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured DelayedHeadStreamingHandler instance
    """
    if config is None:
        config = StreamingConfiguration(**kwargs)
    
    return DelayedHeadStreamingHandler(config=config)