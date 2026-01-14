"""Scaffold for extracting PolicyUnit arrays from retrieved context.

This module provides function signatures and documentation for later integration
with the existing retrieval pipeline and LLM pool.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple
import json

from langchain_core.messages import HumanMessage, SystemMessage

from app.schemas.policy import PolicyUnit
from app.services.validators import parse_json_model
from app.services.prompts_policy import (
    POLICY_UNIT_EXTRACTION_SYSTEM,
    POLICY_UNIT_EXTRACTION_USER,
)


def _get_source_id(metadata: Dict[str, Any]) -> str:
    for key in ("id", "source_id", "document_id", "chunk_id", "parent_id", "uid"):
        value = metadata.get(key)
        if value:
            return str(value)
    return metadata.get("title") or metadata.get("filename") or metadata.get("source") or "unknown"


def _normalize_key(text: str) -> str:
    t = (text or "").strip().lower()
    # simple canonicalization
    for ch in ["/", " ", ",", ":", ";", "(", ")", ".", "|"]:
        t = t.replace(ch, "-")
    while "--" in t:
        t = t.replace("--", "-")
    return t.strip("-")


async def extract_policy_units_from_chunks(
    llm_wrapper,
    chunks: Sequence[Tuple[object, float]] | Sequence[object],
    audience: str,
    max_units: int = 25,
) -> List[PolicyUnit]:
    """Extract normalized policy units from a set of retrieved chunks.

    Parameters
    - chunks: sequence of LangChain Documents (optionally with scores)
    - audience: 'general' or 'classA'
    - max_units: cap to control token/cost usage

    Returns a list of PolicyUnit models extracted by the LLM and validated.
    """
    # Flatten to documents and scores
    docs: List[Any] = []
    if chunks and isinstance(chunks[0], tuple):
        docs = [c[0] for c in chunks]  # type: ignore[index]
    else:
        docs = list(chunks)  # type: ignore[arg-type]

    # Limit the number of chunks included in the prompt
    max_chunks = min(len(docs), 12)
    selected_docs = docs[:max_chunks]

    # Build context block
    sources_block_parts: List[str] = []
    allowed_ids: List[str] = []
    for idx, doc in enumerate(selected_docs):
        metadata = dict(getattr(doc, "metadata", {}) or {})
        source_id = _get_source_id(metadata)
        allowed_ids.append(source_id)
        title = metadata.get("title") or metadata.get("filename") or metadata.get("source") or "Document"
        text = getattr(doc, "page_content", "") or ""
        # Trim very long chunks to keep prompt reasonable
        if len(text) > 1800:
            text = text[:1800] + "..."
        sources_block_parts.append(
            f"ID: {source_id}\nTitle: {title}\nText: {text}"
        )

    messages = [
        SystemMessage(content=POLICY_UNIT_EXTRACTION_SYSTEM),
        HumanMessage(
            content=(
                f"Audience: {audience}\n\n" +
                POLICY_UNIT_EXTRACTION_USER.strip() +
                "\n\nCONTEXT:\n" + "\n\n".join(sources_block_parts) +
                "\n\nReturn JSON only with this shape: {\n  \"units\": [ {\n    \"policyArea\": string, \"dedupeKey\": string, \"subject\": string, \"action\": string,\n    \"conditions\": string[], \"effect\": \"allow|deny|require|limit|n/a\", \"scope\": string|null,\n    \"notes\": string|null, \"citations\": [{\"sourceId\": string, \"anchor\": string|null}], \"audience\": string\n  } ]\n}"
            )
        ),
    ]

    llm = getattr(llm_wrapper, "llm", llm_wrapper)
    response = await llm.ainvoke(messages)
    content = getattr(response, "content", "") or ""

    # Extract JSON
    payload: Dict[str, Any]
    try:
        payload = json.loads(content)
    except Exception:
        # Try to find JSON substring
        start = content.find("{")
        end = content.rfind("}")
        payload = json.loads(content[start : end + 1]) if start >= 0 and end > start else {"units": []}

    raw_units = payload.get("units", []) or []
    units: List[PolicyUnit] = []

    for ru in raw_units[:max_units]:
        try:
            # Force audience and canonicalize dedupeKey
            ru = dict(ru)
            ru["audience"] = audience
            if ru.get("dedupeKey"):
                ru["dedupeKey"] = _normalize_key(str(ru["dedupeKey"]))

            model = parse_json_model(PolicyUnit, ru)
            # Filter invalid citations
            valid_citations = [c for c in (model.citations or []) if c.sourceId in allowed_ids]
            model.citations = valid_citations
            units.append(model)
        except Exception:
            continue

    return units
