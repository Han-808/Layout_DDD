from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


VERDICTS = {"valid", "invalid"}


def adjudicate_candidate(
    *,
    metric: str,
    candidate: dict[str, Any],
    scene: dict[str, Any],
    prompt: str,
    render_evidence: list[str] | None,
    judge: Any,
) -> dict[str, Any]:
    """Request one metric-specific binary judgement.

    The judge may be a callable or expose ``adjudicate_spatial_fidelity``.
    Regardless of transport, a scalar category score is not an
    acceptable response: each routed candidate needs an explicit valid/invalid
    verdict so unresolved infrastructure cannot masquerade as a metric value.
    """

    if judge is None:
        raise RuntimeError("spatial-fidelity adjudication requires a configured VLM judge")
    event = candidate.get("event") if isinstance(candidate.get("event"), dict) else {
        "type": f"{metric}_candidate",
        "object_ids": [],
    }
    object_ids = {
        str(value)
        for value in event.get("object_ids", [])
        if str(value).strip()
    }
    objects = scene.get("objects") if isinstance(scene.get("objects"), list) else []
    request = {
        "category": "spatial_fidelity_adjudication",
        "metric": str(metric),
        "event": deepcopy(event),
        "detector_evidence": deepcopy(candidate),
        "involved_objects": [
            compact_object(obj)
            for obj in objects
            if isinstance(obj, dict) and str(obj.get("id")) in object_ids
        ],
        "required_response": {"verdict": "valid|invalid"},
        "natural_language_prompt": str(prompt or ""),
        "scene_summary": compact_scene(scene),
        "render_evidence": [
            str(Path(path).expanduser())
            for path in render_evidence or []
            if str(path).strip()
        ],
    }
    call = getattr(judge, "adjudicate_spatial_fidelity", None)
    if not callable(call):
        call = judge
    if not callable(call):
        raise TypeError(
            "spatial-fidelity VLM judge must be callable or expose "
            "adjudicate_spatial_fidelity(request)"
        )
    raw = call(request)
    if not isinstance(raw, dict):
        raise ValueError("spatial-fidelity VLM response must be a JSON object")
    verdict = raw.get("verdict")
    if verdict not in VERDICTS:
        raise ValueError("spatial-fidelity VLM verdict must be exactly 'valid' or 'invalid'")
    confidence = raw.get("confidence")
    if confidence is not None:
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError("spatial-fidelity VLM confidence must be numeric")
        confidence = float(confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("spatial-fidelity VLM confidence must be between 0 and 1")
    return {
        "verdict": str(verdict),
        "score": 1.0 if verdict == "valid" else 0.0,
        "confidence": confidence,
        "reason": raw.get("reason"),
        "request": request,
        "judgement": raw,
    }


def compact_scene(scene: dict[str, Any]) -> dict[str, Any]:
    objects = scene.get("objects") if isinstance(scene.get("objects"), list) else []
    return {
        "scene_id": scene.get("scene_id"),
        "request_id": scene.get("request_id"),
        "scene_type": scene.get("scene_type"),
        "boundary": deepcopy(scene.get("boundary")),
        "scene_height": scene.get("scene_height"),
        "objects": [compact_object(obj) for obj in objects if isinstance(obj, dict)],
    }


def compact_object(obj: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": obj.get("id"),
        "category": obj.get("category"),
        "description": obj.get("description") or obj.get("desc") or obj.get("short_desc"),
        "center": deepcopy(obj.get("center")),
        "size": deepcopy(obj.get("size")),
        "rotation_degrees": deepcopy(obj.get("rotation")),
    }


def finalize_metric_report(
    *,
    metric: str,
    evaluator_version: str,
    checks: list[dict[str, Any]],
    eligible_count: int,
    not_applicable_reason: str,
    notes: list[str],
) -> dict[str, Any]:
    resolved = [check for check in checks if is_score(check.get("score"))]
    unknown = [check for check in checks if check.get("route") == "unknown"]
    vlm_pending = [
        check
        for check in checks
        if check.get("route") in {"requires_vlm", "vlm_adjudication_failed"}
        and not is_score(check.get("score"))
    ]
    if eligible_count <= 0:
        status = "not_applicable"
        score = None
        partial_score = None
        reason = not_applicable_reason
    else:
        partial_score = (
            sum(float(check["score"]) for check in resolved) / float(len(resolved))
            if resolved
            else None
        )
        complete = len(resolved) == eligible_count
        status = "checked" if complete else "incomplete"
        score = partial_score if complete else None
        reason = None if complete else "metric_coverage_incomplete"
    route_counts = {
        "direct_valid": sum(check.get("route") == "direct_valid" for check in checks),
        "requires_vlm": sum(check.get("candidate_route") == "requires_vlm" for check in checks),
        "vlm_adjudicated": sum(check.get("route") == "vlm_adjudicated" for check in checks),
        "vlm_adjudication_failed": sum(
            check.get("route") == "vlm_adjudication_failed" for check in checks
        ),
        "unknown": len(unknown),
    }
    return {
        "metric": metric,
        "evaluator_version": evaluator_version,
        "status": status,
        "reason": reason,
        "score": None if score is None else float(score),
        "partial_score": None if partial_score is None else float(partial_score),
        "coverage": {
            "eligible_count": int(max(0, eligible_count)),
            "resolved_count": len(resolved),
            "unknown_count": len(unknown),
            "vlm_pending_count": len(vlm_pending),
            "fraction": (
                float(len(resolved)) / float(eligible_count)
                if eligible_count > 0
                else None
            ),
            "complete": bool(eligible_count > 0 and len(resolved) == eligible_count),
        },
        "routing": route_counts,
        "checks": checks,
        "notes": list(notes),
    }


def is_score(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and 0.0 <= float(value) <= 1.0
    )
