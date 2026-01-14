"""Prompt templates for policy unit extraction and delta summarization.

These are scaffolds; integration will be performed by the retrieval pipeline.
"""

from __future__ import annotations

POLICY_UNIT_EXTRACTION_SYSTEM = (
    "You are an expert policy analyst. Extract atomic policy rules as JSON."
)

POLICY_UNIT_EXTRACTION_USER = (
    """
Read the provided context and extract atomic policy rules as structured PolicyUnit objects.
Guidelines:
- One atomic rule per unit.
- Keep exact numbers/dates/thresholds; do not round or infer.
- Include at least one citation from the provided context for each unit.
- Propose a stable dedupeKey that uniquely identifies the rule within its policyArea.
- Set the 'audience' field to 'general' or 'classA' based on the context set.

Return strict JSON array under key 'units'.
"""
)

DELTA_SUMMARIZATION_SYSTEM = (
    "You are a careful editor. Rewrite a computed diff graph into concise user-facing bullets."
)

DELTA_SUMMARIZATION_USER = (
    """
You are given matched pairs of baseline and classA PolicyUnits with preliminary change type labels,
plus unmatched units on each side. Produce a strict JSON object with keys:
- stricter, looser, additionalRequirements, exceptions, replacements, notApplicable, additions
Each key maps to an array of DeltaItem objects with fields:
- policyArea, dedupeKey, changeType, summary, citations[]
Rules:
- Only include items supported by at least one Class A citation.
- Keep summaries short and specific (one sentence).
- Preserve exact figures and names from sources.
- Do not restate baseline content unless needed for clarity.
"""
)

