"""SSE stream emitter for chat responses."""

import json
import time
import uuid
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional

from app.core.logging import get_logger
from app.models.query import ChatRequest, ChatResponse, Source
from app.models.query_history import QueryStatus
from app.utils.metrics import compute_quality_metrics

logger = get_logger(__name__)


class StreamEmitter:
    """Emits Server-Sent Events for streaming chat responses."""

    def __init__(
        self,
        perf_monitor: Any,
        query_logger: Any,
        source_repository: Optional[Any] = None,
    ):
        """Initialize stream emitter.

        Args:
            perf_monitor: Performance monitor instance.
            query_logger: Query logger instance.
            source_repository: Optional source repository.
        """
        self.perf_monitor = perf_monitor
        self.query_logger = query_logger
        self.source_repository = source_repository

    def emit_connection(self, connection_id: str) -> str:
        """Emit connection event.

        Args:
            connection_id: Unique connection ID.

        Returns:
            SSE formatted event.
        """
        return f"data: {json.dumps({'type': 'connection', 'id': connection_id})}\n\n"

    def emit_metadata(self, **kwargs) -> str:
        """Emit metadata event.

        Args:
            **kwargs: Metadata key-value pairs.

        Returns:
            SSE formatted event.
        """
        payload = {"type": "metadata", **kwargs}
        return f"data: {json.dumps(payload)}\n\n"

    def emit_retrieval_start(self) -> str:
        """Emit retrieval start event.

        Returns:
            SSE formatted event.
        """
        return f"data: {json.dumps({'type': 'retrieval_start'})}\n\n"

    def emit_retrieval_complete(self, duration: float, count: int) -> str:
        """Emit retrieval complete event.

        Args:
            duration: Duration in seconds.
            count: Number of documents retrieved.

        Returns:
            SSE formatted event.
        """
        return f"data: {json.dumps({'type': 'retrieval_complete', 'duration': duration, 'count': count})}\n\n"

    def emit_sources(self, sources: List[Source]) -> str:
        """Emit sources event.

        Args:
            sources: List of sources.

        Returns:
            SSE formatted event.
        """
        return f"data: {json.dumps({'type': 'sources', 'sources': [s.model_dump() for s in sources]})}\n\n"

    def emit_first_token(self, latency_ms: float) -> str:
        """Emit first token event.

        Args:
            latency_ms: Latency in milliseconds.

        Returns:
            SSE formatted event.
        """
        return f"data: {json.dumps({'type': 'first_token', 'latency': latency_ms})}\n\n"

    def emit_token(self, content: str) -> str:
        """Emit token event.

        Args:
            content: Token content.

        Returns:
            SSE formatted event.
        """
        return f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"

    def emit_complete(self, duration: float) -> str:
        """Emit complete event.

        Args:
            duration: Total duration in seconds.

        Returns:
            SSE formatted event.
        """
        return f"data: {json.dumps({'type': 'complete', 'duration': duration})}\n\n"

    def emit_error(self, message: str) -> str:
        """Emit error event.

        Args:
            message: Error message.

        Returns:
            SSE formatted event.
        """
        return f"data: {json.dumps({'type': 'error', 'message': message})}\n\n"

    def emit_follow_up_questions(self, questions: List[Dict[str, Any]]) -> str:
        """Emit follow-up questions event.

        Args:
            questions: List of follow-up question dicts.

        Returns:
            SSE formatted event.
        """
        return f"data: {json.dumps({'type': 'metadata', 'follow_up_questions': questions})}\n\n"

    def emit_delta(self, delta_payload: Dict[str, Any]) -> str:
        """Emit delta metadata event.

        Args:
            delta_payload: Delta payload dict.

        Returns:
            SSE formatted event.
        """
        return f"data: {json.dumps({'type': 'metadata', 'delta': delta_payload})}\n\n"

    async def stream_cached_response(
        self,
        chat_request: ChatRequest,
        conversation_id: str,
        cached_response: ChatResponse,
        start_time: datetime,
        request_timer: float,
    ) -> AsyncGenerator[str, None]:
        """Stream a cached response.

        Args:
            chat_request: The chat request.
            conversation_id: Conversation ID.
            cached_response: The cached response.
            start_time: Request start time.
            request_timer: Performance timer start.

        Yields:
            SSE formatted events.
        """
        cache_latency_ms = (time.perf_counter() - request_timer) * 1000
        response_seconds = cache_latency_ms / 1000 if cache_latency_ms else 0.0

        # Record metrics
        self.perf_monitor.record_latency("search_latency_ms", 0)
        self.perf_monitor.record_latency("context_build_latency_ms", 0)
        self.perf_monitor.record_latency("answer_generation_latency_ms", cache_latency_ms)
        self.perf_monitor.record_latency("llm_latency_ms", cache_latency_ms)
        self.perf_monitor.record_latency("total_request_latency_ms", cache_latency_ms)
        self.perf_monitor.record_latency("first_token_latency_ms", cache_latency_ms)

        tokens_used = cached_response.tokens_used
        if tokens_used is not None:
            try:
                self.perf_monitor.record_token_usage(
                    str(chat_request.provider), int(tokens_used)
                )
            except (TypeError, ValueError):
                logger.debug("Cached tokens_used not numeric: %s", tokens_used)

        # Coerce sources
        raw_sources = cached_response.sources or []
        sources: List[Source] = []
        for item in raw_sources:
            coerced = self._coerce_source(item)
            if coerced:
                sources.append(coerced)

        yield self.emit_retrieval_start()
        yield self.emit_retrieval_complete(0, len(sources))

        if chat_request.include_sources and sources:
            yield self.emit_sources(sources)

        self.perf_monitor.increment_counter("successful_requests")

        quality_metrics = compute_quality_metrics(
            cached_response.response, sources, None
        )
        for key, value in quality_metrics.items():
            self.perf_monitor.record_value(key, value)

        yield self.emit_first_token(cache_latency_ms)
        yield self.emit_token(cached_response.response)
        yield self.emit_complete(response_seconds)

        # Log query
        await self._log_cached_query(
            chat_request,
            conversation_id,
            cached_response,
            sources,
            start_time,
            tokens_used,
        )

    async def _log_cached_query(
        self,
        chat_request: ChatRequest,
        conversation_id: str,
        cached_response: ChatResponse,
        sources: List[Source],
        start_time: datetime,
        tokens_used: Optional[int],
    ) -> None:
        """Log a cached query."""
        processing_time = (datetime.utcnow() - start_time).total_seconds()

        metadata = {
            "temperature": chat_request.temperature,
            "max_tokens": chat_request.max_tokens,
            "include_sources": chat_request.include_sources,
            "retrieval_count": len(sources),
            "retrieval_ms": 0,
            "context_build_ms": 0,
            "follow_up_count": 0,
            "cache_hit": True,
        }

        query_id = str(uuid.uuid4())
        model_name = cached_response.model or chat_request.model or "unknown"

        try:
            await self.query_logger.log_query(
                query_id=query_id,
                user_query=chat_request.message,
                provider=str(chat_request.provider),
                model=model_name,
                use_rag=chat_request.use_rag,
                response=cached_response.response,
                sources_count=len(sources),
                processing_time=processing_time,
                tokens_used=tokens_used,
                conversation_id=conversation_id,
                status=QueryStatus.SUCCESS,
                metadata=metadata,
            )
        except Exception as exc:
            logger.warning("Failed to log cached streaming query: %s", exc)

        if self.source_repository and chat_request.include_sources and sources:
            try:
                await self.source_repository.record_query_sources(
                    query_id,
                    [source.model_dump() for source in sources],
                )
            except Exception as repo_error:
                logger.warning("Failed to record cached query sources: %s", repo_error)

    def _coerce_source(self, payload: Any) -> Optional[Source]:
        """Coerce cached source payloads into Source models."""
        if isinstance(payload, Source):
            return payload

        if payload is None:
            return None

        if isinstance(payload, dict):
            return Source(**payload)

        dump_method = getattr(payload, "model_dump", None)
        if callable(dump_method):
            return Source(**dump_method())

        dict_method = getattr(payload, "dict", None)
        if callable(dict_method):
            return Source(**dict_method())

        logger.debug("Unable to coerce cached source payload: %s", type(payload))
        return None

    def coerce_chat_response(self, payload: Any) -> Optional[ChatResponse]:
        """Coerce cached payloads into a ChatResponse model."""
        if isinstance(payload, ChatResponse):
            return payload

        if payload is None:
            return None

        if isinstance(payload, dict):
            return ChatResponse(**payload)

        dump_method = getattr(payload, "model_dump", None)
        if callable(dump_method):
            return ChatResponse(**dump_method())

        dict_method = getattr(payload, "dict", None)
        if callable(dict_method):
            return ChatResponse(**dict_method())

        logger.debug(
            "Unable to coerce cached payload into ChatResponse: %s", type(payload)
        )
        return None
