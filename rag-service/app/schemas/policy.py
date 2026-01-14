"""Pydantic schemas for policy-unit extraction and delta diffing.

These models are scaffolding for a robust, deterministic pipeline that:
- extracts normalized PolicyUnit objects from retrieved context, and
- computes a DeltaResponse that captures only audience-specific differences.

Integration points will be added in a later change.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class Effect(str, Enum):
    allow = "allow"
    deny = "deny"
    require = "require"
    limit = "limit"
    na = "n/a"


class ChangeType(str, Enum):
    stricter = "stricter"
    looser = "looser"
    additional_requirement = "additionalRequirement"
    exception = "exception"
    replacement = "replacement"
    not_applicable = "notApplicable"
    addition = "addition"


class Citation(BaseModel):
    sourceId: str = Field(..., description="Canonical source ID or URI")
    anchor: Optional[str] = Field(None, description="Section/page anchor where applicable")


class PolicyUnit(BaseModel):
    policyArea: str = Field(..., description="Top-level policy area (e.g., leave, pay)")
    dedupeKey: str = Field(..., description="Canonical rule key for matching across audiences")
    subject: str = Field(..., description="Who is affected by the rule")
    action: str = Field(..., description="What is allowed/required/limited")
    conditions: List[str] = Field(default_factory=list, description="Preconditions/thresholds")
    effect: Effect = Field(..., description="Resulting effect of the rule")
    scope: Optional[str] = Field(None, description="Coverage period/amount/applicability scope")
    notes: Optional[str] = Field(None, description="Short clarifier for UI rendering")
    citations: List[Citation] = Field(default_factory=list, description="Supporting citations")
    audience: Optional[str] = Field(
        None,
        description="Audience label for which this unit was extracted (e.g., general, classA)",
    )


class DeltaItem(BaseModel):
    policyArea: str
    dedupeKey: str
    changeType: ChangeType
    summary: str = Field(..., description="Concise, user-facing description of the delta")
    citations: List[str] = Field(
        default_factory=list,
        description="List of sourceIds supporting the delta (must include audience-specific source)",
    )
    baseline: Optional[PolicyUnit] = None
    classA: Optional[PolicyUnit] = None


class DeltaResponse(BaseModel):
    stricter: List[DeltaItem] = Field(default_factory=list)
    looser: List[DeltaItem] = Field(default_factory=list)
    additionalRequirements: List[DeltaItem] = Field(default_factory=list)
    exceptions: List[DeltaItem] = Field(default_factory=list)
    replacements: List[DeltaItem] = Field(default_factory=list)
    notApplicable: List[DeltaItem] = Field(default_factory=list)
    additions: List[DeltaItem] = Field(default_factory=list)
    debug: Optional[dict] = Field(default=None, description="Optional debugging payload for dev")


__all__ = [
    "Effect",
    "ChangeType",
    "Citation",
    "PolicyUnit",
    "DeltaItem",
    "DeltaResponse",
]

