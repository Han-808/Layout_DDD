from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Iterable

from benchmark.visual_judge.visual_config import (
    DEFAULT_P0B_VISUAL_CONFIGS,
    compose_default_p0b_visual_evidence,
    is_metric_focus_evidence,
)


P0B_METRICS = {"collision", "object_architecture_penetration", "oob", "support"}
P0B_VERDICTS = {"valid", "invalid"}
HIGH_RECALL_CANDIDATE_POLICY = "high_recall_candidate_no_label_prior"
# Metric-specific aliases (identical value) kept for readability at call sites.
COLLISION_CANDIDATE_SELECTION_POLICY = HIGH_RECALL_CANDIDATE_POLICY
SUPPORT_CANDIDATE_SELECTION_POLICY = HIGH_RECALL_CANDIDATE_POLICY
CANDIDATE_SELECTION_POLICY_METRICS = {"collision", "support"}
P0B_METRIC_RUBRICS = {
    "collision": (
        "Judge only whether these two objects have an actual unintended physical surface interpenetration. "
        "The pair was proposed by a high-recall candidate detector; being selected carries no verdict prior "
        "and is not itself evidence that the pair is invalid. "
        "Return invalid only when the supplied evidence supports real, unintended surface interpenetration. "
        "If the evidence does not establish penetration, return valid for collision. "
        "A generator relationship claim is semantic context only: it is neither proof of a collision nor an "
        "automatic exemption. "
        "If reliable mesh evidence reports definitive non-intersection and positive separation, do not return "
        "invalid solely because the OBB proxies overlap or the placement looks awkward. "
        "When mesh evidence is degraded or unavailable, inspect the images for visible surface penetration; "
        "OBB overlap alone is insufficient for an invalid verdict. "
        "Ordinary contact or near-contact, positive separation, intended containment or assembly, wrong "
        "support, floating, sinking, and an unfollowed spatial relation are not collisions by themselves. "
        "Do not judge support, prompt fidelity, object-architecture penetration, or general visual quality here."
    ),
    "object_architecture_penetration": (
        "Judge whether the measured object-architecture interaction is an invalid penetration or an intentional "
        "attachment/contact. Treat detector geometry as measured evidence, not as a semantic verdict."
    ),
    "oob": (
        "The deterministic detector routed this object because its oriented bounding box crosses one or more "
        "flagged room planes beyond the applicable threshold: numerical_eps for the wall and ceiling planes, and "
        "the separate floor_contact_tolerance_m semantic contact tolerance for the floor plane. Treat the reported "
        "plane flags, room bounds, object intervals, and measured crossing depths as authoritative facts; do not "
        "reinterpret, round away, or dispute them. Being routed here is a request for adjudication and carries no "
        "verdict prior: a measured crossing is not by itself sufficient to return invalid. Judge whether this "
        "object substantively protrudes outside the room envelope in a way that is not a permitted structural "
        "condition. Return invalid only when the evidence shows a real, unintended out-of-bounds protrusion. "
        "Return valid when the crossing is an explicitly permitted structural condition, such as a genuinely "
        "wall-embedded, wall-mounted, ceiling-mounted, or pass-through element whose attachment mode and geometry "
        "are compatible with crossing that plane, or when the apparent crossing is ordinary surface contact or "
        "placement tolerance rather than a substantive protrusion. Shallow floor sink within "
        "floor_contact_tolerance_m is already accepted deterministically and is never routed here. A "
        "natural-language or extracted relationship is a claim, not proof of the exception. In particular, "
        "near_wall, against_wall, at_corner, near_corner, along_wall, and room_region never exempt ordinary "
        "furniture or decor from remaining inside the room. Ordinary contact with a wall is not boundary crossing. "
        "Judge only OOB here; do not use prompt fidelity, style, or general visual plausibility to excuse or "
        "manufacture the measured crossing."
    ),
    "support": (
        "Judge only whether an object has an unexplained positive support gap along gravity, or is instead "
        "supported, attached, suspended, or intentionally floating. The object was proposed by a high-recall "
        "detector; being routed here carries no invalid prior. Invalid means only an unexplained positive "
        "support gap. One reliable contact to the floor, or to an object whose full support chain is certified or "
        "visibly grounded, is enough to establish non-floating support; local contact with a floating or ungrounded "
        "support is not. Inspect the entire support chain and any inherited positive gap. A low contact fraction, "
        "an empty center ray, sparse legs/feet/frames, and support split across multiple grounded targets are not errors. "
        "A negative gap, sinking, object-object penetration, and out-of-bounds crossing are never Support-invalid "
        "and belong to Collision/OOB; an object may be Collision-invalid while remaining Support-valid, and both "
        "may be invalid only with independent positive-gap evidence. Do not judge whether the object used the "
        "prompt-requested support target; that belongs to OOR/OAR. Prompt relationships may explain attachment "
        "or suspension but are claims, not proof. Return invalid only for an unexplained positive support gap."
    ),
}

# How to read collision camera evidence. Sent to the judge so the raw/overlay
# pairing and diagnostic colors are unambiguous and geometry stays authoritative.
COLLISION_EVIDENCE_STYLE_GUIDE = (
    "Collision evidence may include paired same-pose views that share a pair_id/view_id: first the raw RGB "
    "render, then a deterministic segmentation-contour view of the exact same camera pose. The contour view preserves "
    "the target interiors and scene appearance; object A and object B receive distinct exterior color bands and thin "
    "outer contours from occlusion-aware Blender object-ID masks. Thin wireframes may outline each target's oriented "
    "bounding box, and yellow may mark the closest/contact point or focus region. These bands, contours, wireframes, "
    "markers, and colors are diagnostic annotations, not physically rendered appearance. Object IDs, categories, "
    "colors, and representation types are provided as structured metadata. Treat the exact detector and "
    "mesh measurements as authoritative evidence and never invent or overwrite them; use the images to judge "
    "whether the measured overlap or intersection is an unintended physical collision, and return one binary verdict."
)

LocalViewProvider = Callable[
    [dict[str, Any]],
    Iterable[str | Path | dict[str, Any]],
]


def adjudicate_p0b_event(
    *,
    metric: str,
    event: dict,
    prompt: str,
    relationships: list[dict] | dict | None,
    scene: dict,
    detector_evidence: dict,
    judge: Any,
    object_ids: list[str] | tuple[str, ...] | None = None,
    overview_render_evidence: list[str] | None = None,
    local_view_provider: LocalViewProvider | None = None,
    visual_config_policy: str = "metric_default",
) -> dict:
    """Resolve one ambiguous P0b event with a mandatory binary VLM verdict.

    The local-view callable is intentionally injected. Camera selection and
    Blender implementation are outside this interface contract.
    """

    metric_name = str(metric)
    if metric_name not in P0B_METRICS:
        raise ValueError(f"unsupported P0b metric {metric_name!r}")
    if not isinstance(event, dict) or not isinstance(detector_evidence, dict):
        raise TypeError("P0b event and detector_evidence must be JSON objects")
    if judge is None:
        raise RuntimeError("P0b VLM adjudication requires a configured judge")
    if visual_config_policy not in {"metric_default", "passthrough"}:
        raise ValueError("visual_config_policy must be 'metric_default' or 'passthrough'")

    resolved_ids = _event_object_ids(event, object_ids)
    objects = [
        _compact_object(item)
        for item in scene.get("objects", [])
        if isinstance(item, dict) and str(item.get("id")) in resolved_ids
    ]
    architecture = {
        "boundary": deepcopy(scene.get("boundary")),
        "height": scene.get("scene_height"),
        "floor_z": 0.0,
        "elements": ["floor", "walls", "ceiling"],
    }
    local_request = build_p0b_local_evidence_request(
        metric=metric_name,
        event=event,
        prompt=prompt,
        relationships=relationships,
        scene=scene,
        detector_evidence=detector_evidence,
        object_ids=resolved_ids,
    )
    local_items = list(local_view_provider(local_request)) if local_view_provider is not None else []
    local_paths, local_metadata = _normalize_local_view_items(local_items)
    overview_paths = list(overview_render_evidence or [])
    applied_visual_config: dict[str, Any] | None = None
    if (
        visual_config_policy == "metric_default"
        and metric_name in DEFAULT_P0B_VISUAL_CONFIGS
        and is_metric_focus_evidence(local_metadata, metric=metric_name)
    ):
        selected_items, applied_visual_config = compose_default_p0b_visual_evidence(
            metric_name,
            local_metadata,
        )
        local_paths, local_metadata = _normalize_local_view_items(selected_items)
        render_evidence = list(local_paths)
        max_images = getattr(judge, "max_images", None)
        if isinstance(max_images, int) and max_images < len(render_evidence):
            raise RuntimeError(
                f"judge max_images={max_images} is below the {metric_name} default "
                f"VisualConfig budget={len(render_evidence)}"
            )
    else:
        # Legacy/path-only providers and frozen experiment arms retain their
        # original evidence ordering. OOB is global-first in that contract.
        render_evidence = _deduplicate_paths(
            overview_paths + local_paths
            if metric_name == "oob"
            else local_paths + overview_paths
        )
    request = {
        "category": "p0b_structural_adjudication",
        "metric": metric_name,
        "metric_rubric": P0B_METRIC_RUBRICS[metric_name],
        "event": deepcopy(event),
        "natural_language_prompt": str(prompt),
        "extracted_relationships": deepcopy(relationships),
        "objects": objects,
        "architecture": architecture,
        "detector_evidence": deepcopy(detector_evidence),
        "local_render_evidence": local_paths,
        "render_evidence": render_evidence,
    }
    if applied_visual_config is not None:
        request["visual_evidence_policy"] = applied_visual_config
    elif metric_name == "oob":
        request["visual_evidence_policy"] = {
            "default_bundle": "fixed_global_plus_deterministic_local",
            "image_order": ["fixed_global", "deterministic_local"],
            "fixed_global_view_count": len(overview_paths),
            "deterministic_local_view_count": len(local_paths),
            "local_camera_mode": "visibility_ranked",
            "pose_selector": "deterministic",
        }
    if metric_name in CANDIDATE_SELECTION_POLICY_METRICS:
        # Collision and Support candidates are proposed by high-recall detectors;
        # selection carries no verdict prior. Surface this explicitly so it
        # reaches the model context.
        request["candidate_selection_policy"] = str(
            event.get("candidate_selection_policy") or HIGH_RECALL_CANDIDATE_POLICY
        )
    if metric_name == "collision":
        request["collision_evidence_style_guide"] = COLLISION_EVIDENCE_STYLE_GUIDE
    if local_metadata:
        request["local_render_evidence_metadata"] = local_metadata
    call = getattr(judge, "adjudicate_p0b", judge)
    if not callable(call):
        raise TypeError("P0b judge must be callable or expose adjudicate_p0b(request)")
    raw = call(request)
    if not isinstance(raw, dict):
        raise ValueError("P0b judge response must be a JSON object")
    verdict = raw.get("verdict")
    if verdict not in P0B_VERDICTS:
        raise ValueError("P0b judge verdict must be exactly 'valid' or 'invalid'")
    confidence = _confidence(raw.get("confidence"))
    return {
        "status": "evaluated",
        "metric": metric_name,
        "verdict": verdict,
        "score": 1.0 if verdict == "valid" else 0.0,
        "confidence": confidence,
        "reason": raw.get("reason"),
        "visual_evidence_policy": deepcopy(request.get("visual_evidence_policy")),
        "request": request,
        "judgement": raw,
    }


def build_p0b_local_evidence_request(
    *,
    metric: str,
    event: dict,
    prompt: str,
    relationships: list[dict] | dict | None,
    scene: dict,
    detector_evidence: dict,
    object_ids: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Build the frozen read-only request consumed by camera providers.

    Keeping this request builder separate lets an offline render phase create
    camera evidence without constructing or calling a VLM judge. The same
    contract remains used by normal online adjudication.
    """

    metric_name = str(metric)
    if metric_name not in P0B_METRICS:
        raise ValueError(f"unsupported P0b metric {metric_name!r}")
    if not isinstance(event, dict) or not isinstance(detector_evidence, dict):
        raise TypeError("P0b event and detector_evidence must be JSON objects")
    resolved_ids = _event_object_ids(event, object_ids)
    return {
        "metric": metric_name,
        "event": deepcopy(event),
        "scene": deepcopy(scene),
        "object_ids": resolved_ids,
        "architecture_element": event.get("architecture_element"),
        "detector_evidence": deepcopy(detector_evidence),
        "natural_language_prompt": str(prompt),
        "extracted_relationships": deepcopy(relationships),
        "access": "read_only_evidence_request",
    }


def _normalize_local_view_items(items: list[Any]) -> tuple[list[str], list[dict[str, Any]]]:
    """Accept both legacy path items and richer evidence dicts.

    Path-only providers keep returning ``str``/``Path`` items and remain fully
    compatible. A provider may instead return dicts carrying optional metadata
    (``role``, ``view_id``, object A/B IDs, color legend, representation level,
    visibility statistics, degradation reason); the file path is always preserved
    as a plain string in ``local_render_evidence`` so existing callers are
    unaffected, and the metadata is exposed alongside it.
    """

    paths: list[str] = []
    metadata: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            raw_path = item.get("path")
            if raw_path is None:
                raise ValueError("local view evidence object must include a 'path'")
            resolved = str(Path(str(raw_path)).expanduser())
            paths.append(resolved)
            entry = {key: deepcopy(value) for key, value in item.items()}
            entry["path"] = resolved
            metadata.append(entry)
        else:
            paths.append(str(Path(str(item)).expanduser()))
    return paths, metadata


def _event_object_ids(event: dict, explicit: list[str] | tuple[str, ...] | None) -> list[str]:
    values: list[Any] = list(explicit or [])
    if not values:
        for key in ["object_id", "object_a", "object_b", "subject_id", "supporting_object_id"]:
            if event.get(key) is not None:
                values.append(event[key])
        if isinstance(event.get("object_ids"), list):
            values.extend(event["object_ids"])
    return list(dict.fromkeys(str(value) for value in values if str(value)))


def _compact_object(obj: dict) -> dict:
    asset_proxy = obj.get("asset_proxy") if isinstance(obj.get("asset_proxy"), dict) else {}
    return {
        "id": obj.get("id"),
        "category": obj.get("category") or obj.get("retrieval_category"),
        "description": obj.get("description") or obj.get("desc") or obj.get("short_desc"),
        "center": deepcopy(obj.get("center")),
        "size": deepcopy(obj.get("size") or asset_proxy.get("bbox_size")),
        "rotation_degrees": deepcopy(obj.get("rotation")),
        "geometry_provenance": obj.get("geometry_provenance"),
    }


def _deduplicate_paths(paths: list[str]) -> list[str]:
    return list(dict.fromkeys(str(Path(path).expanduser()) for path in paths))


def _confidence(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("P0b judge confidence must be numeric")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError("P0b judge confidence must be between 0 and 1")
    return result
