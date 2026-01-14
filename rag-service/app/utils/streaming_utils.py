"""Helpers for interpreting streaming payloads from LangChain/LLM clients."""

from __future__ import annotations

from typing import Any, Iterable, Optional


def coerce_to_text(payload: Any) -> str:
    """Safely coerce streaming payload structures into plain text."""

    if payload is None:
        return ""

    if isinstance(payload, str):
        return payload

    if isinstance(payload, (int, float)):
        return str(payload)

    if isinstance(payload, dict):
        # Handle Gemini 3.0 structured chunks
        if payload.get("type") == "text" and "text" in payload:
            return str(payload["text"])

        for key in ("text", "output_text", "content", "delta", "message"):
            if key in payload:
                text_value = coerce_to_text(payload[key])
                if text_value:
                    return text_value
        fragments = [coerce_to_text(value) for value in payload.values()]
        return "".join(fragment for fragment in fragments if fragment)

    for attr in ("content", "text", "output_text", "delta", "message"):
        if hasattr(payload, attr):
            text_value = coerce_to_text(getattr(payload, attr))
            if text_value:
                return text_value

    if hasattr(payload, "additional_kwargs"):
        # Message-like payload with no textual delta yet
        return ""

    if isinstance(payload, Iterable):
        fragments = [coerce_to_text(item) for item in payload]
        return "".join(fragment for fragment in fragments if fragment)

    return ""


def extract_chunk_text(chunk: Any) -> str:
    """Extract textual content from a LangChain chunk object."""

    if chunk is None:
        return ""

    if isinstance(chunk, str):
        return chunk

    return coerce_to_text(chunk)


def extract_token_usage_from_chunk(chunk: Any) -> Optional[int]:
    """Best-effort extraction of token usage from a streaming chunk."""

    if chunk is None:
        return None

    candidate_maps = []
    for attr in ("generation_info", "response_metadata", "metadata", "info"):
        value = getattr(chunk, attr, None)
        if isinstance(value, dict):
            candidate_maps.append(value)

    if isinstance(chunk, dict):
        candidate_maps.append(chunk)

    for mapping in candidate_maps:
        for key in ("token_usage", "usage", "usage_metadata"):
            usage = mapping.get(key)
            if isinstance(usage, dict):
                for token_key in ("total_tokens", "total", "totalTokens", "completion_tokens"):
                    token_value = usage.get(token_key)
                    if isinstance(token_value, (int, float)):
                        return int(token_value)

    return None

