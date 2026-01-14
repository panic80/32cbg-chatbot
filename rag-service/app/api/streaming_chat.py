"""Streaming chat API endpoint that streams LLM tokens as they are generated."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from datetime import datetime
from typing import AsyncGenerator, Dict, List, Optional, Tuple, Any

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage

from app.api.chat import get_llm
from app.api.chat import generate_followup
from app.api.prompt_constants import (
    CHAT_SYSTEM_PROMPT,
    GMT_GLOSSARY_NOTE,
    NO_CONTEXT_PROMPT_TEMPLATE,
    SHORT_ANSWER_PROMPT,
    TRIP_PLAN_INSTRUCTION,
)
from app.core.config import settings
from app.core.logging import get_logger
from app.models.query import ChatRequest, ChatResponse, Provider, Source, FollowUpRequest
from app.models.query_history import QueryStatus
from app.services.advanced_cache import AdvancedCacheService, create_context_hash
from app.services.llm_pool import llm_pool
from app.services.performance_monitor import get_performance_monitor
from app.services.query_logger import get_query_logger
from app.utils.message_utils import build_history_messages
from app.utils.metrics import compute_quality_metrics
from app.utils.streaming_utils import (
    extract_chunk_text,
    extract_token_usage_from_chunk,
)
from app.services.policy_units import extract_policy_units_from_chunks
from app.services.policy_diff import match_units, build_delta
from app.services.policy_delta_summarizer import summarize_delta_with_llm
from app.components.hyde_generator import get_hyde_generator
from app.services.model_selector import get_model_selector

# Import modular services
from app.services.chat.query_processor import (
    QueryProcessor,
    ensure_provider,
    resolve_model,
    should_use_hybrid,
    build_classification_note,
)
from app.services.chat.metadata_enricher import MetadataEnricher
from app.services.chat.retrieval_executor import RetrievalExecutor
from app.services.chat.response_builder import ResponseBuilder
from app.services.chat.stream_emitter import StreamEmitter

logger = get_logger(__name__)
router = APIRouter()

ENTITLEMENT_DENIAL_DIRECTIVE = (
    "Heuristic assessment indicates the member is likely NOT entitled to the requested allowance. "
    "Unless the retrieved documentation explicitly proves entitlement, you must:\n"
    "- State clearly and directly that the meal entitlement is not authorized for the described scenario.\n"
    "- Cite the governing policy conditions (e.g., TD/tasking requirements, orders outside normal hours) that are missing.\n"
    "- Explain what circumstances would change the outcome (such as being on TD or having written orders).\n"
    "- Encourage the member to confirm with their chain of command or DCBA if they believe additional directives apply.\n"
    "Do not provide narratives that imply the entitlement can be claimed when the policy conditions are not met."
)


async def _stream_events(
    request: Request,
    chat_request: ChatRequest,
) -> AsyncGenerator[str, None]:
    """Stream chat response events."""
    perf_monitor = get_performance_monitor()
    query_logger = get_query_logger()

    perf_monitor.increment_counter("streaming_requests")
    perf_monitor.increment_counter("total_requests")

    start_time = datetime.utcnow()
    request_timer = time.perf_counter()

    connection_id = str(uuid.uuid4())
    yield f"data: {json.dumps({'type': 'connection', 'id': connection_id})}\n\n"

    conversation_id = chat_request.conversation_id or str(uuid.uuid4())
    yield f"data: {json.dumps({'type': 'metadata', 'conversation_id': conversation_id})}\n\n"

    provider_enum = ensure_provider(chat_request.provider)
    model_name = resolve_model(provider_enum, chat_request.model)

    vector_store_manager = getattr(request.app.state, "vector_store_manager", None)
    if vector_store_manager is None:
        raise RuntimeError("Vector store manager is not configured")

    cache_service = getattr(request.app.state, "cache_service", None)
    advanced_cache = AdvancedCacheService(cache_service) if cache_service else None
    source_repository = getattr(request.app.state, "source_repository", None)

    llm_wrapper = None

    try:
        try:
            async with llm_pool.acquire(provider_enum, model_name) as pooled_llm:
                llm_wrapper = pooled_llm
                async for event in _run_streaming_flow(
                    request=request,
                    chat_request=chat_request,
                    conversation_id=conversation_id,
                    llm_wrapper=llm_wrapper,
                    vector_store=vector_store_manager,
                    app_state=request.app.state,
                    perf_monitor=perf_monitor,
                    start_time=start_time,
                    request_timer=request_timer,
                    advanced_cache=advanced_cache,
                    source_repository=source_repository,
                    query_logger=query_logger,
                ):
                    yield event
            return
        except Exception as pool_error:
            logger.warning("LLM pool acquisition failed (%s). Falling back to direct client.", pool_error)
            llm_wrapper = await asyncio.to_thread(get_llm, provider_enum, model_name)
            async for event in _run_streaming_flow(
                request=request,
                chat_request=chat_request,
                conversation_id=conversation_id,
                llm_wrapper=llm_wrapper,
                vector_store=vector_store_manager,
                app_state=request.app.state,
                perf_monitor=perf_monitor,
                start_time=start_time,
                request_timer=request_timer,
                advanced_cache=advanced_cache,
                source_repository=source_repository,
                query_logger=query_logger,
            ):
                yield event
    except asyncio.CancelledError:
        logger.info("Streaming request cancelled by client")
        perf_monitor.increment_counter("failed_requests")
        yield f"data: {json.dumps({'type': 'error', 'message': 'Connection cancelled'})}\n\n"
    except Exception as exc:
        logger.exception("Streaming chat failed: %s", exc)
        perf_monitor.increment_counter("failed_requests")
        yield f"data: {json.dumps({'type': 'error', 'message': 'Streaming temporarily unavailable'})}\n\n"


async def _run_streaming_flow(
    request: Request,
    chat_request: ChatRequest,
    conversation_id: str,
    llm_wrapper,
    vector_store,
    app_state,
    perf_monitor,
    start_time: datetime,
    request_timer: float,
    advanced_cache: Optional[AdvancedCacheService],
    source_repository: Any,
    query_logger: Any,
) -> AsyncGenerator[str, None]:
    """Run the main streaming flow."""
    if await request.is_disconnected():
        return

    provider_enum = ensure_provider(chat_request.provider)
    resolved_model_name = resolve_model(provider_enum, chat_request.model)
    requested_model = (chat_request.model or resolved_model_name or "").strip().lower()
    # Fast mode = optimized path (skip HyDE, limited context, skip classification)
    # Smart mode = full pipeline (thorough retrieval)
    # Mode is determined by frontend based on user's selection in Config page
    is_fast_mode = chat_request.mode == "fast"
    logger.info(f"Request mode: {chat_request.mode}, is_fast_mode: {is_fast_mode}, model: {requested_model}")

    # Emit retrieval_start early in fast mode for better UX
    if is_fast_mode and chat_request.use_rag:
        yield f"data: {json.dumps({'type': 'retrieval_start'})}\n\n"

    # Determine auxiliary model using ModelSelector and overrides
    model_selector = get_model_selector()
    aux_provider, aux_model = model_selector.get_model_for_operation("retrieval")
    
    # Allow per-request override
    if chat_request.retrieval_config and chat_request.retrieval_config.auxiliary_model:
        aux_model = chat_request.retrieval_config.auxiliary_model
        # Assume same provider for now, or default to OpenAI if unknown
        # Ideally RetrievalConfig should support provider override too
        
    auxiliary_model = aux_model
    auxiliary_provider = aux_provider

    logger.info(f"Using auxiliary model: {auxiliary_model} (Provider: {auxiliary_provider.value})")

    # Process query - skip LLM acquisition in fast mode since classification is skipped
    classification_start = time.perf_counter()
    classification_data = None

    if is_fast_mode:
        # Fast mode: skip LLM acquisition entirely, just expand abbreviations
        query_processor = QueryProcessor(None)
        classification_data = await query_processor.process_query(
            chat_request.message, is_fast_mode=True
        )
    else:
        # Smart mode: use auxiliary model (gpt-5-mini) for classification
        try:
            async with llm_pool.acquire(auxiliary_provider, auxiliary_model) as classifier_llm:
                query_processor = QueryProcessor(classifier_llm)
                classification_data = await query_processor.process_query(
                    chat_request.message, is_fast_mode=False
                )
        except Exception as e:
            logger.warning(f"Classification failed ({e}). Falling back to main model.")
            query_processor = QueryProcessor(llm_wrapper)
            classification_data = await query_processor.process_query(
                chat_request.message, is_fast_mode=False
            )

    optimized_query, classification, classification_dict = classification_data
    
    classification_time_ms = (time.perf_counter() - classification_start) * 1000
    perf_monitor.record_latency("query_classification_latency_ms", classification_time_ms)

    classification_note = build_classification_note(classification_dict)
    entitlement_denial_required = bool(
        classification_dict and classification_dict.get("entitlement_likely_denied")
    )

    # Initialize services with main LLM (for later use if needed)
    # Note: QueryProcessor was already used, but we might need it again for other things? 
    # Actually, process_query is the main thing.
    
    optimizer_llm = None if is_fast_mode else llm_wrapper
    # Re-init processor just in case other methods are called later with main LLM
    query_processor = QueryProcessor(optimizer_llm) 
    metadata_enricher = MetadataEnricher()
    retrieval_executor = RetrievalExecutor(vector_store, app_state, llm_wrapper)
    response_builder = ResponseBuilder()
    stream_emitter = StreamEmitter(perf_monitor, query_logger, source_repository)

    # Initialize HyDE generator (skip for smart mode to reduce latency)
    # Check for per-request enable_hyde override
    hyde_generator = None
    enable_hyde = settings.enable_hyde
    if chat_request.retrieval_config and chat_request.retrieval_config.enable_hyde is not None:
        enable_hyde = chat_request.retrieval_config.enable_hyde
        logger.info(f"Using per-request enable_hyde: {enable_hyde}")
    if enable_hyde and not is_fast_mode:
        cache_service = getattr(app_state, "cache_service", None)
        hyde_generator = get_hyde_generator(llm_pool, cache_service)

    # Initialize state
    retrieval_results: List[Tuple] = []
    follow_up_questions_payload: List[Dict[str, Any]] = []
    context = ""
    sources: List[Source] = []
    retrieval_count = 0
    retrieval_time_ms = 0.0
    context_time_ms = 0.0
    glossary_injected = False
    kilometric_snippet_added = False

    # Check cache
    cached_response: Optional[ChatResponse] = None
    cache_model = chat_request.model or "default"

    if advanced_cache:
        cache_lookup_hash = hashlib.md5(
            f"{chat_request.message}:{chat_request.provider}".encode()
        ).hexdigest()
        try:
            cached_payload = await advanced_cache.get_response(
                query=chat_request.message,
                context_hash=cache_lookup_hash,
                model=cache_model,
            )
        except Exception as cache_error:
            logger.warning("Advanced cache lookup failed: %s", cache_error)
            cached_payload = None

        if cached_payload:
            cached_response = stream_emitter.coerce_chat_response(cached_payload)
            if cached_response:
                perf_monitor.record_cache_hit("l3", True)
                async for event in stream_emitter.stream_cached_response(
                    chat_request=chat_request,
                    conversation_id=conversation_id,
                    cached_response=cached_response,
                    start_time=start_time,
                    request_timer=request_timer,
                ):
                    yield event
                return
            logger.debug("Cached payload not coercible to ChatResponse; treating as miss")
            perf_monitor.record_cache_hit("l3", False)
        else:
            perf_monitor.record_cache_hit("l3", False)

    # Retrieval
    if chat_request.use_rag:
        # Only emit retrieval_start for smart mode (fast mode already emitted it early)
        if not is_fast_mode:
            yield f"data: {json.dumps({'type': 'retrieval_start'})}\n\n"

        # Use auxiliary model (gpt-5-mini) for retrieval (multi-query generation)
        retrieval_llm_wrapper = llm_wrapper
        try:
            async with llm_pool.acquire(auxiliary_provider, auxiliary_model) as retrieval_llm:
                retrieval_executor = RetrievalExecutor(vector_store, app_state, retrieval_llm)

                pipeline = await retrieval_executor.create_pipeline(chat_request, is_fast_mode)
                retrieval_start = time.perf_counter()
                
                # Use concurrent HyDE execution (pass generator, not hypothesis string)
                retrieval_results = await retrieval_executor.retrieve(
                    pipeline, 
                    optimized_query, 
                    is_fast_mode=is_fast_mode,
                    hyde_generator=hyde_generator,
                    classification=classification_dict,
                    auxiliary_model=auxiliary_model
                )
        except Exception as e:
            logger.warning(f"Auxiliary model retrieval failed ({e}). Falling back to main model.")
            retrieval_executor = RetrievalExecutor(vector_store, app_state, llm_wrapper)
            pipeline = await retrieval_executor.create_pipeline(chat_request, is_fast_mode)
            retrieval_start = time.perf_counter()
            retrieval_results = await retrieval_executor.retrieve(
                pipeline, 
                optimized_query, 
                is_fast_mode=is_fast_mode,
                hyde_generator=hyde_generator,
                classification=classification_dict,
                auxiliary_model=auxiliary_model
            )

        retrieval_time_ms = (time.perf_counter() - retrieval_start) * 1000

        retrieval_count = len(retrieval_results)

        yield f"data: {json.dumps({'type': 'retrieval_complete', 'duration': retrieval_time_ms / 1000, 'count': retrieval_count})}\n\n"
        perf_monitor.record_latency("search_latency_ms", retrieval_time_ms)

        if retrieval_results:
            context_build_start = time.perf_counter()
            context, sources = await response_builder.build_context_and_sources(
                retrieval_results, optimized_query, vector_store
            )

            # Enrich with kilometric rate
            context, sources, kilometric_snippet_added = metadata_enricher.enrich_with_kilometric_rate(
                context, sources, chat_request, optimized_query, classification_dict
            )

            # Enrich with glossary
            context, sources, _, glossary_injected = metadata_enricher.enrich_with_glossary(
                context, sources, chat_request.message
            )

            # Apply smart mode context limit
            if is_fast_mode:
                char_limit = getattr(settings, "smart_mode_context_char_limit", 0)
                context = response_builder.truncate_context(context, char_limit)

            context_time_ms = (time.perf_counter() - context_build_start) * 1000
            perf_monitor.record_latency("context_build_latency_ms", context_time_ms)

            if chat_request.include_sources and sources:
                yield f"data: {json.dumps({'type': 'sources', 'sources': [s.model_dump() for s in sources]})}\n\n"
        else:
            perf_monitor.record_latency("context_build_latency_ms", 0)
    else:
        perf_monitor.record_latency("search_latency_ms", 0)
        perf_monitor.record_latency("context_build_latency_ms", 0)

    # Build messages
    system_prompt = CHAT_SYSTEM_PROMPT
    if chat_request.additional_instructions:
        system_prompt = f"{CHAT_SYSTEM_PROMPT}\n\nADDITIONAL DIRECTIVES:\n{chat_request.additional_instructions.strip()}"

    messages: List[Any] = [SystemMessage(content=system_prompt)]
    if chat_request.short_answer_mode:
        messages.append(SystemMessage(content=SHORT_ANSWER_PROMPT))
    if classification_note:
        messages.append(SystemMessage(content=classification_note))
    if entitlement_denial_required:
        messages.append(SystemMessage(content=ENTITLEMENT_DENIAL_DIRECTIVE))
    messages.extend(build_history_messages(chat_request))

    user_message = chat_request.message
    if chat_request.use_rag:
        user_message = f"{chat_request.message} and show me references"

    if context:
        citation_guide = response_builder.build_citation_guide(sources)
        context_prompt = (
            "Based on the following official documentation, answer the user's question:\n\n"
            f"{context}"
            f"{citation_guide}"
            f"{TRIP_PLAN_INSTRUCTION}\n"
            f"User Question: {user_message}"
        )
        messages.append(HumanMessage(content=context_prompt))
    else:
        if chat_request.use_rag:
            no_context_prompt = NO_CONTEXT_PROMPT_TEMPLATE.format(question=user_message)
            messages.append(HumanMessage(content=no_context_prompt))
        else:
            messages.append(HumanMessage(content=chat_request.message))

    llm = getattr(llm_wrapper, "llm", llm_wrapper)

    # Compute Class A delta - START IN BACKGROUND to avoid blocking TTFT
    delta_task: Optional[asyncio.Task[Optional[dict]]] = None
    if chat_request.use_rag and retrieval_results:
        delta_task = asyncio.create_task(_compute_class_a_delta(llm_wrapper, retrieval_results))

    # Stream generation
    generation_start = time.perf_counter()
    first_token_sent = False
    full_response_parts: List[str] = []
    token_usage_total: Optional[int] = None

    stream_kwargs: Dict[str, Any] = {}
    underlying_llm = getattr(llm, "llm", llm)
    model_name_str = getattr(underlying_llm, "model_name", resolved_model_name)
    model_name_lower = (model_name_str or "").strip().lower()

    if provider_enum == Provider.OPENAI:
        if model_name_lower.startswith("o") and chat_request.reasoning_effort:
            stream_kwargs["reasoning"] = {"effort": chat_request.reasoning_effort}
        if chat_request.response_verbosity:
            if model_name_lower.startswith("o"):
                stream_kwargs.setdefault("reasoning", {})["verbosity"] = chat_request.response_verbosity
            else:
                logger.debug("Skipping response_verbosity for non-reasoning model %s", model_name_str)
        if chat_request.max_tokens:
            stream_kwargs["max_tokens"] = int(chat_request.max_tokens)

    chunk_debug_counter = 0
    async for chunk in llm.astream(messages, **stream_kwargs):
        if await request.is_disconnected():
            logger.info("Client disconnected during generation")
            raise asyncio.CancelledError

        if (
            provider_enum == Provider.OPENAI
            and requested_model == "gpt-5-mini"
            and chunk_debug_counter < 5
            and logger.isEnabledFor(logging.DEBUG)
        ):
            logger.debug("[STREAM_CHUNK_DEBUG] %s", repr(chunk))
            chunk_debug_counter += 1

        token_text = extract_chunk_text(chunk)
        if not token_text:
            continue

        full_response_parts.append(token_text)

        usage_from_chunk = extract_token_usage_from_chunk(chunk)
        if usage_from_chunk is not None:
            token_usage_total = usage_from_chunk

        if not first_token_sent and token_text.strip():
            first_token_sent = True
            latency_ms = (time.perf_counter() - generation_start) * 1000
            perf_monitor.record_latency("first_token_latency_ms", latency_ms)
            yield f"data: {json.dumps({'type': 'first_token', 'latency': latency_ms})}\n\n"

        yield f"data: {json.dumps({'type': 'token', 'content': token_text})}\n\n"

    full_response = "".join(full_response_parts).strip()

    # Append glossary note if needed
    if glossary_injected:
        full_response, appended_text = metadata_enricher.append_glossary_to_response(
            full_response, glossary_injected
        )
        if appended_text:
            yield f"data: {json.dumps({'type': 'token', 'content': appended_text})}\n\n"

    if token_usage_total is not None:
        try:
            perf_monitor.record_token_usage(str(chat_request.provider), int(token_usage_total))
        except (TypeError, ValueError):
            logger.debug("Streaming token usage not numeric: %s", token_usage_total)

    answer_latency_ms = (time.perf_counter() - generation_start) * 1000
    total_latency_ms = (time.perf_counter() - request_timer) * 1000
    perf_monitor.record_latency("answer_generation_latency_ms", answer_latency_ms)
    perf_monitor.record_latency("llm_latency_ms", answer_latency_ms)
    perf_monitor.record_latency("total_request_latency_ms", total_latency_ms)

    # Emit delta if any (wait for the background task to complete)
    if delta_task is not None:
        try:
            delta_payload = await delta_task
            if delta_payload:
                yield f"data: {json.dumps({'type': 'metadata', 'delta': delta_payload})}\n\n"
        except Exception as emit_exc:
            logger.debug("Failed to emit delta metadata: %s", emit_exc)

    quality_metrics = compute_quality_metrics(full_response, sources, retrieval_results)
    for key, value in quality_metrics.items():
        perf_monitor.record_value(key, value)

    # Generate follow-up questions
    followup_task: Optional[asyncio.Task[List[Dict[str, Any]]]] = None
    if chat_request.use_rag and full_response:
        followup_task = asyncio.create_task(
            _generate_followups(request, chat_request, full_response, sources)
        )

    perf_monitor.increment_counter("successful_requests")
    yield f"data: {json.dumps({'type': 'complete', 'duration': total_latency_ms / 1000})}\n\n"

    if followup_task:
        try:
            follow_up_questions_payload = await followup_task
        except Exception as exc:
            logger.warning("Follow-up task failed: %s", exc)
            follow_up_questions_payload = []

        if follow_up_questions_payload:
            yield f"data: {json.dumps({'type': 'metadata', 'follow_up_questions': follow_up_questions_payload})}\n\n"

    # Log query
    await _log_streaming_query(
        query_logger=query_logger,
        source_repository=source_repository,
        chat_request=chat_request,
        conversation_id=conversation_id,
        full_response=full_response,
        sources=sources,
        llm=llm,
        start_time=start_time,
        retrieval_count=retrieval_count,
        retrieval_time_ms=retrieval_time_ms,
        context_time_ms=context_time_ms,
        follow_up_questions_payload=follow_up_questions_payload,
        token_usage_total=token_usage_total,
        classification_dict=classification_dict,
        entitlement_denial_required=entitlement_denial_required,
        glossary_injected=glossary_injected,
        kilometric_snippet_added=kilometric_snippet_added,
    )


async def _compute_class_a_delta(
    llm_wrapper: Any,
    retrieval_results: List[Tuple],
) -> Optional[dict]:
    """Compute Class A policy delta."""
    try:
        def _is_class_a(meta: Dict[str, Any]) -> bool:
            audience_tag = (meta.get("audience") or "").strip().lower()
            title = (meta.get("title") or meta.get("filename") or meta.get("source") or "").lower()
            section = (meta.get("section") or "").lower()
            tags = meta.get("tags") or []
            tags_lower = [str(t).lower() for t in (tags if isinstance(tags, list) else [])]
            hay = " ".join([audience_tag, title, section, " ".join(tags_lower)])
            return ("class a" in hay) or (audience_tag == "classa")

        baseline_docs = []
        class_docs = []
        for doc, _score in retrieval_results:
            meta = dict(getattr(doc, "metadata", {}) or {})
            if _is_class_a(meta):
                class_docs.append(doc)
            else:
                baseline_docs.append(doc)

        if not class_docs:
            return None

        baseline_units = await extract_policy_units_from_chunks(
            llm_wrapper, baseline_docs, audience="general", max_units=25
        )
        class_units = await extract_policy_units_from_chunks(
            llm_wrapper, class_docs, audience="classA", max_units=25
        )

        matches, baseline_only, class_only = match_units(baseline_units, class_units)
        delta = build_delta(matches, baseline_only, class_only)

        # Filter items without Class A citations
        class_ids = set()
        for d in class_docs:
            m = dict(getattr(d, "metadata", {}) or {})
            sid = m.get("id") or m.get("source_id") or m.get("document_id") or m.get("chunk_id")
            if sid:
                class_ids.add(str(sid))

        def _has_class_citation(item) -> bool:
            if not item.citations:
                return False
            return any((c in class_ids) for c in item.citations)

        delta.stricter = [i for i in delta.stricter if _has_class_citation(i)]
        delta.looser = [i for i in delta.looser if _has_class_citation(i)]
        delta.additionalRequirements = [i for i in delta.additionalRequirements if _has_class_citation(i)]
        delta.exceptions = [i for i in delta.exceptions if _has_class_citation(i)]
        delta.replacements = [i for i in delta.replacements if _has_class_citation(i)]
        delta.additions = [i for i in delta.additions if _has_class_citation(i)]

        delta_summarized = await summarize_delta_with_llm(llm_wrapper, delta)
        return delta_summarized.model_dump()

    except Exception as delta_exc:
        logger.warning("Delta computation failed: %s", delta_exc)
        return None


async def _generate_followups(
    request: Request,
    chat_request: ChatRequest,
    full_response: str,
    sources: List[Source],
) -> List[Dict[str, Any]]:
    """Generate follow-up questions."""
    try:
        followup_request = FollowUpRequest(
            user_question=chat_request.message,
            ai_response=full_response,
            sources=sources,
            max_questions=3,
        )
        followup_response = await generate_followup(request, followup_request)
        return [q.model_dump() for q in followup_response.questions]
    except Exception as exc:
        logger.warning("Failed to generate follow-up questions: %s", exc)
        return []


async def _log_streaming_query(
    query_logger: Any,
    source_repository: Any,
    chat_request: ChatRequest,
    conversation_id: str,
    full_response: str,
    sources: List[Source],
    llm: Any,
    start_time: datetime,
    retrieval_count: int,
    retrieval_time_ms: float,
    context_time_ms: float,
    follow_up_questions_payload: List[Dict[str, Any]],
    token_usage_total: Optional[int],
    classification_dict: Optional[Dict[str, Any]],
    entitlement_denial_required: bool,
    glossary_injected: bool,
    kilometric_snippet_added: bool,
) -> None:
    """Log a streaming query."""
    source_ids = [source.source_id or source.id for source in sources] if sources else []
    metadata = {
        "temperature": chat_request.temperature,
        "max_tokens": chat_request.max_tokens,
        "include_sources": chat_request.include_sources,
        "retrieval_count": retrieval_count,
        "retrieval_ms": retrieval_time_ms,
        "context_build_ms": context_time_ms,
        "follow_up_count": len(follow_up_questions_payload),
        "source_ids": source_ids,
        "cache_hit": False,
        "classification": classification_dict,
        "entitlement_denial_hint": entitlement_denial_required,
        "glossary_injected": glossary_injected,
        "kilometric_snippet_added": kilometric_snippet_added,
    }

    query_id = str(uuid.uuid4())

    try:
        await query_logger.log_query(
            query_id=query_id,
            user_query=chat_request.message,
            provider=str(chat_request.provider),
            model=chat_request.model or getattr(llm, "model_name", "unknown"),
            use_rag=chat_request.use_rag,
            response=full_response,
            sources_count=len(sources),
            processing_time=(datetime.utcnow() - start_time).total_seconds(),
            tokens_used=token_usage_total,
            conversation_id=conversation_id,
            status=QueryStatus.SUCCESS,
            metadata=metadata,
        )
    except Exception as log_exc:
        logger.warning("Failed to log streaming query: %s", log_exc)

    if source_repository and sources:
        try:
            await source_repository.record_query_sources(
                query_id,
                [source.model_dump() for source in sources],
            )
        except Exception as repo_error:
            logger.warning("Failed to record query sources: %s", repo_error)


@router.post("/chat/stream")
async def streaming_chat_endpoint(request: Request, chat_request: ChatRequest) -> StreamingResponse:
    """Return an SSE stream for chat responses."""
    return StreamingResponse(_stream_events(request, chat_request), media_type="text/event-stream")


@router.post("/streaming_chat")
async def streaming_chat_legacy(request: Request, chat_request: ChatRequest) -> StreamingResponse:
    """Legacy endpoint maintained for existing gateway integration."""
    return await streaming_chat_endpoint(request, chat_request)


class RetrievalOnlyRequest(BaseModel):
    """Request model for retrieval-only endpoint."""
    query: str = Field(..., description="Search query")
    retrieval_config: Optional[Dict[str, Any]] = Field(None, description="Retrieval config overrides")
    use_hybrid_search: bool = Field(True, description="Enable hybrid search")


class RetrievalOnlyResponse(BaseModel):
    """Response model for retrieval-only endpoint."""
    sources: List[Dict[str, Any]] = Field(..., description="Retrieved sources")
    latency_ms: float = Field(..., description="Retrieval latency in milliseconds")
    retrieval_count: int = Field(..., description="Number of sources retrieved")


@router.post("/retrieval", response_model=RetrievalOnlyResponse)
async def retrieval_only_endpoint(request: Request, retrieval_request: RetrievalOnlyRequest) -> RetrievalOnlyResponse:
    """Fast retrieval-only endpoint - returns sources without LLM generation.

    Useful for evaluation and testing retrieval configurations.
    """
    import time
    from app.models.query import ChatRequest, RetrievalConfig

    start_time = time.perf_counter()

    vector_store_manager = getattr(request.app.state, "vector_store_manager", None)
    if vector_store_manager is None:
        raise RuntimeError("Vector store manager is not configured")

    # Build a minimal ChatRequest to use the retrieval executor
    retrieval_config = None
    if retrieval_request.retrieval_config:
        retrieval_config = RetrievalConfig(**retrieval_request.retrieval_config)

    chat_request = ChatRequest(
        message=retrieval_request.query,
        use_rag=True,
        use_hybrid_search=retrieval_request.use_hybrid_search,
        retrieval_config=retrieval_config,
    )

    # Initialize retrieval executor
    retrieval_executor = RetrievalExecutor(vector_store_manager, request.app.state, None)
    response_builder = ResponseBuilder()

    # Check for per-request enable_hyde override
    enable_hyde = settings.enable_hyde
    if retrieval_config and retrieval_config.enable_hyde is not None:
        enable_hyde = retrieval_config.enable_hyde

    # Initialize HyDE generator if enabled
    hyde_generator = None
    if enable_hyde:
        cache_service = getattr(request.app.state, "cache_service", None)
        hyde_generator = get_hyde_generator(llm_pool, cache_service)

    auxiliary_model = None
    if retrieval_config and retrieval_config.auxiliary_model:
        auxiliary_model = retrieval_config.auxiliary_model

    # Create pipeline and retrieve
    pipeline = await retrieval_executor.create_pipeline(chat_request, is_fast_mode=False)
    
    # Use concurrent HyDE execution
    retrieval_results = await retrieval_executor.retrieve(
        pipeline, 
        retrieval_request.query, 
        is_fast_mode=False, 
        hyde_generator=hyde_generator,
        auxiliary_model=auxiliary_model
    )

    # Build sources
    sources = []
    if retrieval_results:
        _, sources_list = await response_builder.build_context_and_sources(
            retrieval_results, retrieval_request.query, vector_store_manager
        )
        sources = [s.model_dump() for s in sources_list]

    latency_ms = (time.perf_counter() - start_time) * 1000

    return RetrievalOnlyResponse(
        sources=sources,
        latency_ms=latency_ms,
        retrieval_count=len(sources),
    )


