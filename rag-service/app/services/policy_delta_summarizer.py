"""Summarize raw delta into concise bullets using the LLM.

This module rewrites DeltaResponse items by filling the `summary` fields while
preserving existing structure and citations.
"""

from __future__ import annotations

import json
from typing import Any
from pydantic import ValidationError
from langchain_core.messages import HumanMessage, SystemMessage

from app.schemas.policy import DeltaResponse
from app.services.validators import parse_json_model
from app.services.prompts_policy import DELTA_SUMMARIZATION_SYSTEM, DELTA_SUMMARIZATION_USER


async def summarize_delta_with_llm(llm_wrapper, delta: DeltaResponse) -> DeltaResponse:
    """Ask the LLM to produce concise summaries per delta item.

    The model must output the same JSON shape (DeltaResponse). We reuse citations
    exactly as provided and only fill/change the `summary` fields.
    """
    if not delta:
        return delta

    # Prepare a compact payload with only fields the LLM needs
    compact = delta.model_dump()

    system = SystemMessage(content=DELTA_SUMMARIZATION_SYSTEM)
    user = HumanMessage(
        content=(
            DELTA_SUMMARIZATION_USER.strip()
            + "\n\nRAW_DELTA_JSON:\n"
            + json.dumps(compact, ensure_ascii=False)
            + "\n\nIMPORTANT:\n- Do not modify citations arrays.\n- Keep the same keys/categories.\n- Only fill concise `summary` fields.\n- Return a strict JSON object."
        )
    )

    llm = getattr(llm_wrapper, "llm", llm_wrapper)
    resp = await llm.ainvoke([system, user])
    content = getattr(resp, "content", "") or ""

    try:
        data = json.loads(content)
    except Exception:
        start = content.find("{")
        end = content.rfind("}")
        data = json.loads(content[start : end + 1]) if start >= 0 and end > start else compact

    try:
        summarized = parse_json_model(DeltaResponse, data)
        return summarized
    except (ValidationError, Exception):
        # Fall back to original delta if parsing failed
        return delta

