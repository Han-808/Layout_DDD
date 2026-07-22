from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from benchmark.relation_identity import copy_relation_identity


RELATION_FAMILIES = {"oor", "oar"}
RELATION_VERDICTS = {"valid", "invalid"}


def adjudicate_unsupported_relation(
    *,
    family: str,
    relation: dict,
    prompt: str,
    scene: dict,
    render_evidence: list[str] | None,
    judge: Any,
    detector_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Judge one explicit relation that deterministic evidence could not safely resolve."""

    family_name = str(family).strip().lower()
    if family_name not in RELATION_FAMILIES:
        raise ValueError(f"unsupported relationship family {family_name!r}")
    if not isinstance(relation, dict):
        raise TypeError("relationship claim must be a JSON object")
    if judge is None:
        raise RuntimeError("unsupported relationship adjudication requires a configured VLM judge")
    paths = [str(Path(path).expanduser()) for path in render_evidence or [] if str(path).strip()]
    if not paths:
        raise RuntimeError("unsupported relationship adjudication requires rendered visual evidence")

    involved_ids = _relation_object_ids(relation)
    involved_objects = [
        _compact_object(obj)
        for obj in scene.get("objects", []) if isinstance(scene.get("objects"), list)
        if isinstance(obj, dict) and str(obj.get("id")) in involved_ids
    ]
    request = {
        "category": "relationship_fidelity_adjudication",
        "family": family_name,
        "relation": deepcopy(relation),
        "natural_language_prompt": str(prompt or ""),
        "involved_objects": involved_objects,
        "scene_summary": {
            "scene_id": scene.get("scene_id"),
            "scene_type": scene.get("scene_type"),
            "boundary": deepcopy(scene.get("boundary")),
            "scene_height": scene.get("scene_height"),
            "objects": [
                _compact_object(obj)
                for obj in scene.get("objects", []) if isinstance(scene.get("objects"), list)
                if isinstance(obj, dict)
            ],
        },
        "render_evidence": paths,
    }
    if isinstance(detector_evidence, dict):
        request["detector_evidence"] = deepcopy(detector_evidence)
    call = getattr(judge, "adjudicate_relation", judge)
    if not callable(call):
        raise TypeError("relationship VLM judge must be callable or expose adjudicate_relation(request)")
    raw = call(request)
    if not isinstance(raw, dict):
        raise ValueError("relationship VLM judge response must be a JSON object")
    verdict = raw.get("verdict")
    if verdict not in RELATION_VERDICTS:
        raise ValueError("relationship VLM verdict must be exactly 'valid' or 'invalid'")
    confidence = _confidence(raw.get("confidence"))
    return copy_relation_identity({
        "relation": str(relation.get("type") or relation.get("relation") or ""),
        "family": family_name,
        "category": "vlm_fallback",
        "subject_id": relation.get("subject_id"),
        "subject_ids": deepcopy(relation.get("subject_ids")),
        "object_id": relation.get("object_id"),
        "object_ids": deepcopy(relation.get("object_ids")),
        "passed": verdict == "valid",
        "score": 1.0 if verdict == "valid" else 0.0,
        "status": "checked",
        "backend": "vlm",
        "evidence": {
            "reason": raw.get("reason"),
            "confidence": confidence,
            "request": request,
            "judgement": raw,
        },
    }, relation)


def pending_relation_result(
    *,
    family: str,
    relation: dict,
    reason: str,
    error: str | None = None,
    detector_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    evidence: dict[str, Any] = {"reason": reason}
    if error:
        evidence["error"] = str(error)
    if isinstance(detector_evidence, dict):
        evidence["detector_evidence"] = deepcopy(detector_evidence)
    return copy_relation_identity({
        "relation": str(relation.get("type") or relation.get("relation") or ""),
        "family": str(family),
        "category": "vlm_fallback",
        "subject_id": relation.get("subject_id"),
        "subject_ids": deepcopy(relation.get("subject_ids")),
        "object_id": relation.get("object_id"),
        "object_ids": deepcopy(relation.get("object_ids")),
        "passed": None,
        "score": None,
        "status": "requires_vlm" if reason != "vlm_adjudication_failed" else "vlm_adjudication_failed",
        "backend": "vlm",
        "evidence": evidence,
    }, relation)


def _relation_object_ids(relation: dict) -> set[str]:
    values: list[Any] = []
    for key in ("subject_id", "object_id", "anchor_id", "target_id"):
        if relation.get(key) is not None:
            values.append(relation[key])
    for key in ("subject_ids", "object_ids", "member_ids"):
        if isinstance(relation.get(key), list):
            values.extend(relation[key])
    return {str(value) for value in values if str(value).strip()}


def _compact_object(obj: dict) -> dict[str, Any]:
    asset_proxy = obj.get("asset_proxy") if isinstance(obj.get("asset_proxy"), dict) else {}
    return {
        "id": obj.get("id"),
        "category": obj.get("category") or obj.get("retrieval_category"),
        "description": obj.get("description") or obj.get("desc") or obj.get("short_desc"),
        "center": deepcopy(obj.get("center")),
        "size": deepcopy(obj.get("size") or asset_proxy.get("bbox_size")),
        "rotation_degrees": deepcopy(obj.get("rotation")),
        "asset_key": (obj.get("asset_ref") or {}).get("asset_key"),
    }


def _confidence(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("relationship VLM confidence must be numeric")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError("relationship VLM confidence must be between 0 and 1")
    return result
