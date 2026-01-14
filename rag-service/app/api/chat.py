"""Chat API router (streaming-only service)."""

from __future__ import annotations

import asyncio
import re
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request, status
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.core.logging import get_logger
from app.models.query import (
    ChatRequest,
    ChatResponse,
    FollowUpQuestion,
    FollowUpRequest,
    FollowUpResponse,
    Provider,
)
from app.utils.langchain_utils import RetryableLLM

router = APIRouter()
logger = get_logger(__name__)


def normalize_followup_question(text: str, max_length: int = 80) -> str:
    """Trim filler phrases, collapse whitespace, and enforce a short question."""
    if not text:
        return text

    trimmed = re.sub(
        r"^(?:please|kindly)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    trimmed = re.sub(
        r"^(?:can|could|would|will|do)\s+you\s+(?:please\s+)?",
        "",
        trimmed,
        flags=re.IGNORECASE,
    )
    trimmed = re.sub(r"\s+", " ", trimmed).strip()
    trimmed = trimmed.rstrip(".! ")

    if len(trimmed) > max_length:
        words = trimmed.split()
        shortened_words = []
        total_length = 0
        for word in words:
            proposed = total_length + len(word) + (1 if shortened_words else 0)
            if proposed >= max_length:
                break
            shortened_words.append(word)
            total_length = proposed
        if shortened_words:
            trimmed = " ".join(shortened_words)
        else:
            trimmed = trimmed[: max_length].rstrip()

    if not trimmed.endswith("?"):
        trimmed = trimmed.rstrip("?") + "?"

    return trimmed


def get_llm(provider: Provider, model: Optional[str] = None) -> RetryableLLM:
    """Create a deterministic LLM client for the given provider/model."""

    if isinstance(provider, str):
        provider = Provider(provider)

    if provider == Provider.OPENAI:
        if not settings.openai_api_key:
            raise ValueError("OpenAI API key not configured")

        model_name = model or settings.openai_chat_model
        model_name_lower = (model_name or "").strip().lower()

        # Check if it's a reasoning model (for max_tokens handling)
        is_reasoning_model = bool(model_name_lower) and (
            model_name_lower.startswith("o1")
            or model_name_lower.startswith("o3")
            or model_name_lower.startswith("o4")
        )

        logger.info("Creating OpenAI LLM for model: %s", model_name)

        llm_kwargs: Dict[str, Any] = {
            "api_key": settings.openai_api_key,
            "model": model_name,
        }

        if is_reasoning_model or model_name_lower == "gpt-5-mini":
            llm_kwargs["max_tokens"] = 8192

        llm = ChatOpenAI(**llm_kwargs)
        return RetryableLLM(llm)

    if provider in (Provider.GOOGLE, Provider.ANTHROPIC):
        raise ValueError(
            f"Provider '{provider}' is temporarily disabled to enforce deterministic retrieval."
        )

    raise ValueError(f"Unsupported provider: {provider}")


@router.post("/followup", response_model=FollowUpResponse)
async def generate_followup(
    request: Request,  # pylint: disable=unused-argument
    followup_request: FollowUpRequest,
) -> FollowUpResponse:
    """Generate follow-up questions for a completed chat exchange."""

    try:
        llm = await asyncio.to_thread(get_llm, Provider.OPENAI)

        sources_text = ""
        if followup_request.sources:
            sources_text = "\n\nBased on these sources:\n" + "\n".join(
                [
                    f"- {s.title or 'Document'}: {s.text[:100]}..."
                    for s in followup_request.sources[:3]
                ]
            )

        prompt = (
            f"Based on this conversation, generate {followup_request.max_questions} relevant "
            "follow-up questions that help the user continue the discussion:\n\n"
            f"User Question: \"{followup_request.user_question}\"\n"
            f"AI Response: \"{followup_request.ai_response}\""
            f"{sources_text}\n\n"
            "Guidelines:\n"
            "1. Keep each question sharply focused on the key details above.\n"
            "2. Limit every question to 12 words and no more than 80 characters.\n"
            "3. Avoid filler phrases such as \"Could you\" or \"Please\".\n"
            "4. Balance clarification, related insights, and practical next steps.\n\n"
            "Format each question on its own line beginning with \"Q:\"."
        )

        response = await llm.ainvoke([HumanMessage(content=prompt)])

        questions = []
        lines = response.content.split("\n")

        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith("Q:"):
                question_text = line[2:].strip()
            elif line and not line.startswith("Q:") and i > 0:
                question_text = line
            else:
                continue

            if not question_text:
                continue

            question_text = normalize_followup_question(question_text)
            if not question_text:
                continue

            category = "general"
            lowered = question_text.lower()
            if any(word in lowered for word in ["how", "steps", "process"]):
                category = "procedural"
            elif any(word in lowered for word in ["why", "reason", "purpose"]):
                category = "explanatory"
            elif any(word in lowered for word in ["when", "deadline", "time"]):
                category = "temporal"

            questions.append(
                FollowUpQuestion(
                    id=f"followup_{uuid.uuid4().hex[:8]}",
                    question=question_text,
                    category=category,
                    confidence=0.7,
                )
            )

            if len(questions) >= followup_request.max_questions:
                break

        if not questions:
            fallback_definitions = [
                (
                    "followup_default_1",
                    "Need an example that fits this situation?",
                    "clarification",
                ),
                (
                    "followup_default_2",
                    "Which requirements apply to this scenario?",
                    "requirements",
                ),
                (
                    "followup_default_3",
                    "Where is the official reference for this?",
                    "resources",
                ),
            ]
            questions = [
                FollowUpQuestion(
                    id=fallback_id,
                    question=normalize_followup_question(fallback_text),
                    category=fallback_category,
                    confidence=0.5,
                )
                for fallback_id, fallback_text, fallback_category in fallback_definitions
            ]

        return FollowUpResponse(questions=questions[: followup_request.max_questions])

    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Follow-up generation failed: %s", exc, exc_info=True)
        return FollowUpResponse(
            questions=[
                FollowUpQuestion(
                    id="followup_error",
                    question=normalize_followup_question("Need more detail on that?"),
                    category="clarification",
                    confidence=0.3,
                )
            ]
        )
