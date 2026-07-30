from __future__ import annotations

from typing import Any

from benchmark.visual_judge.roles import (
    DecisionContract,
    VLMRole,
    vlm_audit_metadata,
)


def evaluate_vlm_category(
    *,
    category: str,
    prompt: str | None,
    scene: dict,
    render_evidence: list[str],
    judge: Any | None,
    deterministic_evidence: dict | None = None,
) -> dict:
    """Evaluate one VLM-primary category through an injected judge interface."""

    if not render_evidence:
        return _unavailable(category, "missing_standardized_renders")
    if judge is None:
        return _unavailable(category, "vlm_judge_not_configured")
    request = {
        "category": category,
        "prompt": prompt,
        "scene_summary": {
            "scene_id": scene.get("scene_id"),
            "scene_type": scene.get("scene_type"),
            "boundary": scene.get("boundary"),
            "scene_height": scene.get("scene_height"),
            "object_count": len(scene.get("objects", [])) if isinstance(scene.get("objects"), list) else 0,
            "objects": [_compact_object(item) for item in scene.get("objects", []) if isinstance(item, dict)],
        },
        "render_evidence": list(render_evidence),
        "deterministic_evidence": deterministic_evidence,
        **vlm_audit_metadata(
            VLMRole.JUDGE,
            decision_contract=DecisionContract.GENERIC_VISUAL_SCORE,
            judge_method="evaluate",
        ),
    }
    call = getattr(judge, "evaluate", judge)
    if not callable(call):
        raise TypeError("vlm_judge must be callable or expose evaluate(request)")
    raw = call(request)
    if not isinstance(raw, dict):
        raise ValueError("vlm_judge response must be a JSON object")
    if raw.get("applicable") is False:
        return {
            "status": "not_evaluable",
            "score": None,
            "vlm_policy": "primary",
            "request": request,
            "reason": "vlm_judge_insufficient_evidence",
            "judgement": raw,
        }
    score = raw.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0.0 <= float(score) <= 1.0:
        raise ValueError("vlm_judge response score must be between 0 and 1")
    return {
        "status": "evaluated",
        "score": float(score),
        "vlm_policy": "primary",
        "request": request,
        "judgement": raw,
    }


def _compact_object(obj: dict) -> dict:
    metadata = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}
    asset_proxy = obj.get("asset_proxy") if isinstance(obj.get("asset_proxy"), dict) else {}
    return {
        "id": obj.get("id"),
        "category": obj.get("category") or obj.get("retrieval_category"),
        "description": obj.get("description") or obj.get("desc") or obj.get("short_desc"),
        "center": obj.get("center"),
        "size": obj.get("size") or asset_proxy.get("bbox_size"),
        "rotation_degrees": obj.get("rotation"),
        "asset_key": (obj.get("asset_ref") or {}).get("asset_key"),
        "asset_resolution": metadata.get("asset_resolution"),
        "proxy_type": asset_proxy.get("type"),
    }


def _unavailable(category: str, reason: str) -> dict:
    return {
        "status": "not_evaluable",
        "score": None,
        "vlm_policy": "primary",
        "category": category,
        "reason": reason,
    }
