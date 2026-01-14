"""Scaffold for deterministic policy diff computation.

Matches PolicyUnit arrays and classifies differences for an audience (e.g., Class A).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

from app.schemas.policy import ChangeType, DeltaItem, DeltaResponse, PolicyUnit


def _index_by_key(units: Sequence[PolicyUnit]) -> Dict[Tuple[str, str], PolicyUnit]:
    return {(u.policyArea, u.dedupeKey): u for u in units}


def _tokenize(text: str) -> List[str]:
    return [t for t in (text or "").lower().replace("/", " ").split() if t]


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def match_units(
    baseline: Sequence[PolicyUnit], class_a: Sequence[PolicyUnit], similarity_threshold: float = 0.8
) -> Tuple[List[Tuple[PolicyUnit, PolicyUnit]], List[PolicyUnit], List[PolicyUnit]]:
    """Match baseline units to classA units by (policyArea, dedupeKey) then fuzzy fallback.

    Returns (matches, baseline_only, classA_only).
    """
    matches: List[Tuple[PolicyUnit, PolicyUnit]] = []
    baseline_only: List[PolicyUnit] = []
    class_only = list(class_a)

    by_key = _index_by_key(class_a)
    consumed_keys: set[Tuple[str, str]] = set()

    # First pass: exact key match
    for b in baseline:
        key = (b.policyArea, b.dedupeKey)
        c = by_key.get(key)
        if c is not None:
            matches.append((b, c))
            consumed_keys.add(key)
    class_only = [c for c in class_a if (c.policyArea, c.dedupeKey) not in consumed_keys]

    # Second pass: fuzzy match within same policyArea using Jaccard of action+conditions
    for b in baseline:
        key = (b.policyArea, b.dedupeKey)
        if key in consumed_keys:
            continue
        b_tokens = _tokenize(" ".join([b.action] + list(b.conditions)))
        best: Tuple[float, int] | None = None
        best_idx: int | None = None
        for idx, c in enumerate(class_only):
            if c.policyArea != b.policyArea:
                continue
            c_tokens = _tokenize(" ".join([c.action] + list(c.conditions)))
            sim = _jaccard(b_tokens, c_tokens)
            if best is None or sim > best[0]:
                best = (sim, idx)
                best_idx = idx
        if best and best[0] >= similarity_threshold and best_idx is not None:
            matches.append((b, class_only[best_idx]))
            class_only.pop(best_idx)
        else:
            baseline_only.append(b)

    return matches, baseline_only, class_only


def classify_diff(b: PolicyUnit, c: PolicyUnit) -> ChangeType:
    """Heuristic classification for scaffolding purposes.

    Real implementation should compare effect, scope, and conditions.
    """
    # Very coarse placeholders to be refined later
    if b.effect != c.effect:
        # allow -> deny/require => stricter; deny/require -> allow => looser
        if (b.effect == "allow" and c.effect in ("deny", "require")) or (
            b.effect == "limit" and c.effect in ("deny", "require")
        ):
            return ChangeType.stricter
        if (c.effect == "allow" and b.effect in ("deny", "require")) or (
            c.effect == "limit" and b.effect in ("deny", "require")
        ):
            return ChangeType.looser
        return ChangeType.replacement

    # If effects equal, use conditions count as a weak signal
    if len(c.conditions) > len(b.conditions):
        return ChangeType.additional_requirement
    if len(c.conditions) < len(b.conditions):
        return ChangeType.looser
    return ChangeType.replacement


def build_delta(
    matches: Sequence[Tuple[PolicyUnit, PolicyUnit]],
    baseline_only: Sequence[PolicyUnit],
    class_only: Sequence[PolicyUnit],
) -> DeltaResponse:
    """Construct a DeltaResponse from matched and unmatched units.

    This implementation is conservative and primarily scaffolding.
    """
    response = DeltaResponse()

    # Matched pairs
    for b, c in matches:
        change = classify_diff(b, c)
        item = DeltaItem(
            policyArea=b.policyArea,
            dedupeKey=b.dedupeKey,
            changeType=change,
            summary="",  # to be summarized by LLM later
            citations=[cit.sourceId for cit in (c.citations or []) if cit.sourceId],
            baseline=b,
            classA=c,
        )
        if change == ChangeType.stricter:
            response.stricter.append(item)
        elif change == ChangeType.looser:
            response.looser.append(item)
        elif change == ChangeType.additional_requirement:
            response.additionalRequirements.append(item)
        elif change == ChangeType.exception:
            response.exceptions.append(item)
        else:
            response.replacements.append(item)

    # Unmatched
    for b in baseline_only:
        response.notApplicable.append(
            DeltaItem(
                policyArea=b.policyArea,
                dedupeKey=b.dedupeKey,
                changeType=ChangeType.not_applicable,
                summary="",
                citations=[],
                baseline=b,
            )
        )
    for c in class_only:
        response.additions.append(
            DeltaItem(
                policyArea=c.policyArea,
                dedupeKey=c.dedupeKey,
                changeType=ChangeType.addition,
                summary="",
                citations=[cit.sourceId for cit in (c.citations or []) if cit.sourceId],
                classA=c,
            )
        )

    return response

