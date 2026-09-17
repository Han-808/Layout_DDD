"""Trusted identity and stage scopes for opt-in Placement handoffs."""
from copy import deepcopy
from typing import Any


PLACEMENT_SCOPE_INSTRUCTION = """Placement scope contract v2: existence, context and
scoring ownership are separate. Context IDs may name real objects outside the
active group, but are never scoring owners. Same-call judge-originated final
findings must belong to the active stage and active subject scope. scene_zone
always belongs to scene_global; a contextual_anchor across groups also belongs
to scene_global. Do not return either as a group-local final defect. If a
cross-stage observation needs review, use evidence_request.metadata.
placement_check_proposal with its original check_type, subject_id, context_ids
and observation_goal. Include its subject_id in evidence_request.target_ids.
The controller will defer it to its rightful stage; it is not a verdict here.
Do not change a type, subject, context, or conclusion merely to pass validation.
If the current scope itself remains unsupported, retain ambiguous with its
missing observations. Never turn context objects into scored targets."""


def scene_ids(request: dict[str, Any]) -> set[str]:
    for key in ("camera_scene_context", "scene_summary", "scene_context"):
        scene = request.get(key)
        if not isinstance(scene, dict):
            continue
        rows = scene.get("objects") or []
        ids = [str(row.get("id") or row.get("object_id") or "")
               for row in rows if isinstance(row, dict)]
        if ids:
            if any(not value for value in ids) or len(ids) != len(set(ids)):
                raise ValueError("Placement requires unique authoritative scene IDs")
            return set(ids)
    raise ValueError("Placement has no authoritative scene identities")


def groups_for_request(request: dict[str, Any]) -> list[dict[str, Any]]:
    return deepcopy(request.get("placement_scene_groups") or request.get("object_groups") or [])


def validate_subject_scope(request: dict[str, Any], subject_id: str) -> None:
    scope = request.get("group_scope")
    if isinstance(scope, dict) and subject_id not in set(scope.get("member_ids") or []):
        raise ValueError("Placement proposal subject is outside the active scoring group")
    scope = request.get("target_scope")
    if isinstance(scope, dict):
        ids = ([scope["target_id"]] if scope.get("target_id")
               else scope.get("target_ids") or scope.get("member_ids"))
        if ids and subject_id not in set(ids):
            raise ValueError("Placement proposal subject is outside the active scoring target")
