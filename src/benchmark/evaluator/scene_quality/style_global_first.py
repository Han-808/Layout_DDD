"""Shared routing helper for structured screens that select local groups.

Style no longer has a conditional global-screen implementation. Its active
runtime uses ``global_group_first`` so global decisions never short-circuit
eligible group-local review. The remaining helper is shared by the distinct
JSON-screen routes for scale and object pairing.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def suspicious_groups(
    groups: list[dict[str, Any]],
    *,
    judgement: dict[str, Any],
    include_all_when_unlocalized: bool,
) -> list[dict[str, Any]]:
    """Map Judge-localized objects/groups onto immutable grouping scopes."""

    target_ids: set[str] = {
        str(target_id)
        for defect in judgement.get("defects") or []
        if isinstance(defect, dict)
        for target_id in defect.get("target_ids") or []
        if str(target_id)
    }
    group_ids: set[str] = set()
    evidence_request = judgement.get("evidence_request")
    if isinstance(evidence_request, dict):
        target_ids.update(
            str(target_id)
            for target_id in evidence_request.get("target_ids") or []
            if str(target_id) and str(target_id) != "scene"
        )
        metadata = evidence_request.get("metadata")
        if isinstance(metadata, dict):
            group_ids.update(
                str(group_id)
                for group_id in metadata.get("suspicious_group_ids") or []
                if str(group_id)
            )
    selected = [
        deepcopy(group)
        for group in groups
        if str(group.get("group_id")) in group_ids
        or bool(
            target_ids.intersection(
                str(member) for member in group.get("object_ids") or []
            )
        )
    ]
    if selected or not include_all_when_unlocalized:
        return selected
    return [deepcopy(group) for group in groups]
