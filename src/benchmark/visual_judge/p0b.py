from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Iterable

from benchmark.architecture_policy import architecture_contract_from_scene
from benchmark.evaluator.context_projection import (
    project_scene_for_evaluator_context,
)
from benchmark.visual_judge.visual_config import (
    DEFAULT_P0B_VISUAL_CONFIGS,
    compose_default_p0b_visual_evidence,
    is_metric_focus_evidence,
)
from benchmark.visual_judge.roles import (
    DecisionContract,
    VLMRole,
    vlm_audit_metadata,
)


P0B_METRICS = {"collision", "object_architecture_penetration", "oob", "support"}
P0B_VERDICTS = {"valid", "invalid"}
_SUPPORT_LOCAL_RAW_ROLES = {"metric_local_rgb", "collision_rgb"}
P0B_JUDGE_CONTEXT_POLICY_VERSION = (
    "p0b_single_room_baseline_plus_physical_walls_v2"
)
_PHYSICAL_WALL_METRICS = {
    "collision",
    "object_architecture_penetration",
    "oob",
    "support",
}
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
        "A shallow support-interface overlap with a thin horizontal surface layer is not automatically invalid: "
        "independent rigid meshes may encode a compliant covering and its load-bearing object at the same substrate "
        "height even though the real layer would compress. When structured shallow-surface-layer evidence is present, "
        "return valid only if the images and object semantics support ordinary compression or embedding of a compliant "
        "layer, the overlap stays within the reported bounded depth, and the other object does not cross the supporting "
        "substrate. This is not a generic tolerance for small intersections. Return invalid for a rigid layer, lateral "
        "or mid-body slicing, penetration beyond the bounded layer interface, substrate crossing, or visibly impossible "
        "geometry. The structured candidate carries no valid prior. "
        "Do not judge support, prompt fidelity, room-envelope violations, or general visual quality here."
    ),
    "object_architecture_penetration": (
        "Judge whether the measured object-architecture interaction is an invalid penetration or an intentional "
        "attachment/contact. Treat detector geometry as measured evidence, not as a semantic verdict."
    ),
    "oob": (
        "The deterministic detector routed this object because its oriented bounding box crosses one or more "
        "flagged room-envelope planes beyond the applicable threshold. Active physical walls use their room-facing "
        "inner surfaces; inactive or open sides use the logical room boundary. numerical_eps applies to wall and "
        "ceiling planes, and "
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

    scene = project_scene_for_evaluator_context(scene)
    nonrect_scene = _is_nonrect_scene(scene)

    resolved_ids = _event_object_ids(event, object_ids)
    objects = [
        _compact_object(item)
        for item in scene.get("objects", [])
        if isinstance(item, dict) and str(item.get("id")) in resolved_ids
    ]
    architecture = _observable_architecture_for_scene(scene)
    local_request = build_p0b_local_evidence_request(
        metric=metric_name,
        event=event,
        prompt=prompt,
        relationships=relationships,
        scene=scene,
        detector_evidence=detector_evidence,
        object_ids=resolved_ids,
    )
    local_acquisition_exhaustion: dict[str, Any] | None = None
    try:
        local_items = (
            list(local_view_provider(local_request))
            if local_view_provider is not None
            else []
        )
    except Exception as exc:
        if not nonrect_scene:
            raise
        from benchmark.non_rectangular.camera import (
            NonRectangularCameraEvidenceExhausted,
        )

        if not isinstance(
            exc,
            NonRectangularCameraEvidenceExhausted,
        ):
            raise
        local_items = []
        local_acquisition_exhaustion = {
            "error_type": type(exc).__name__,
            "reason": "bounded_nonrect_local_camera_search_exhausted",
        }
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
        "judge_context_policy_version": P0B_JUDGE_CONTEXT_POLICY_VERSION,
        "metric_rubric": P0B_METRIC_RUBRICS[metric_name],
        "event": _project_event_for_judge(metric_name, event),
        "objects": objects,
        "architecture": _project_architecture_for_judge(
            metric_name,
            architecture,
        ),
        "detector_evidence": _project_detector_evidence_for_judge(
            metric_name,
            detector_evidence,
        ),
        "local_render_evidence": local_paths,
        "render_evidence": render_evidence,
        # Additive compatibility input for a later bounded evidence round.
        # The provider receives the same frozen read-only request as the
        # initial acquisition; the metric Judge never consumes this field.
        "camera_evidence_request": deepcopy(local_request),
        **vlm_audit_metadata(
            VLMRole.JUDGE,
            decision_contract=DecisionContract.P0B_BINARY,
            judge_method="adjudicate_p0b",
        ),
    }
    if str(prompt or "").strip():
        request["natural_language_prompt"] = str(prompt)
        if relationships is not None:
            request["extracted_relationships"] = deepcopy(relationships)
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
    local_metric_view_count = _metric_local_view_count(
        metric=metric_name,
        local_paths=local_paths,
        local_metadata=local_metadata,
    )
    nonrect_local_exhausted = bool(
        nonrect_scene
        and visual_config_policy == "metric_default"
        and local_metric_view_count == 0
    )
    if nonrect_local_exhausted:
        provider_usage = getattr(local_view_provider, "last_call_usage", None)
        candidate_audit = (
            deepcopy(provider_usage.get("nonrect_candidate_audit"))
            if isinstance(provider_usage, dict)
            and isinstance(provider_usage.get("nonrect_candidate_audit"), dict)
            else None
        )
        request["nonrect_evidence_continuity"] = {
            "schema_version": "nonrect_p0b_evidence_continuity_v1",
            "policy": "retained_visual_plus_geometry_forced_binary",
            "metric": metric_name,
            "local_visual_count": 0,
            "retained_visual_count": len(render_evidence),
            "geometry_context_available": True,
            "candidate_generation_audit": candidate_audit,
            "typed_acquisition_exhaustion": deepcopy(
                local_acquisition_exhaustion
            ),
            "degraded": True,
        }
        request["visual_evidence_policy"] = {
            "schema_version": "nonrect_p0b_visual_fallback_v1",
            "policy": "retained_global_plus_geometry_forced_choice",
            "image_order": ["retained_global_or_prior"],
            "local_view_count": 0,
            "retained_visual_count": len(render_evidence),
            "selection_source": "bounded_nonrect_camera_exhaustion",
        }
        request["budget_exhaustion_finalization"] = {
            "required": True,
            "trigger_stop_reason": (
                f"nonrect_{metric_name}_local_camera_exhausted"
            ),
            "ambiguity_before_forcing": True,
            "visual_evidence_policy": (
                "retained_global_plus_geometry_forced_choice"
            ),
            "available_visual_count": len(render_evidence),
            "previous_missing_observations": [
                "metric_local_view",
            ],
            "previous_evidence_request": {
                "target_ids": list(resolved_ids),
                "missing_observations": ["metric_local_view"],
                "view_goal": (
                    "The bounded non-rectangular local camera search was "
                    "exhausted. Make the most educated binary choice from "
                    "retained visuals and authoritative geometry; absence of "
                    "a local image carries no valid or invalid prior."
                ),
                "metadata": {
                    "evaluation_mode": "non_rectangular_multi_room",
                    "geometry_only_allowed": True,
                },
            },
        }
    elif (
        metric_name == "support"
        and visual_config_policy == "metric_default"
        and _support_local_raw_count(
            local_paths=local_paths,
            local_metadata=local_metadata,
        ) == 0
    ):
        # Support remains a mandatory binary metric even when camera
        # acquisition yields no local raw view. Reuse the controller's existing
        # forced-choice path with all other available evidence and deterministic
        # context; do not create an unresolved/partial result from view count.
        request["budget_exhaustion_finalization"] = {
            "required": True,
            "trigger_stop_reason": "support_zero_local_evidence",
            "ambiguity_before_forcing": False,
            "visual_evidence_policy": (
                "all_available_then_judge_context_bounded"
            ),
            "available_visual_count": len(render_evidence),
            "previous_missing_observations": [],
            "previous_evidence_request": None,
        }
    runtime_judge = _with_p0b_evidence_control(
        judge,
        local_view_provider=local_view_provider,
    )
    raw_nonrect_forced_call = (
        getattr(judge, "_adjudicate_p0b_raw", None)
        if nonrect_local_exhausted
        else None
    )
    call = (
        raw_nonrect_forced_call
        if callable(raw_nonrect_forced_call)
        else getattr(runtime_judge, "adjudicate_p0b", runtime_judge)
    )
    if not callable(call):
        raise TypeError("P0b judge must be callable or expose adjudicate_p0b(request)")
    raw = (
        call(request, _allow_need_more_evidence=False)
        if callable(raw_nonrect_forced_call)
        else call(request)
    )
    if not isinstance(raw, dict):
        raise ValueError("P0b judge response must be a JSON object")
    verdict = raw.get("verdict")
    if verdict not in P0B_VERDICTS:
        raise ValueError("P0b judge verdict must be exactly 'valid' or 'invalid'")
    confidence = _confidence(raw.get("confidence"))
    if nonrect_local_exhausted:
        forced = request["budget_exhaustion_finalization"]
        raw = deepcopy(raw)
        raw["budget_exhaustion_forced_choice"] = {
            "applied": True,
            "trigger": forced["trigger_stop_reason"],
            "ambiguity_before_forcing": bool(
                forced.get("ambiguity_before_forcing")
            ),
            "pre_force_judge_status": "not_invoked_without_local_evidence",
            "pre_force_evidence_request": deepcopy(
                forced.get("previous_evidence_request")
            ),
            "pre_force_reason": "bounded_nonrect_local_camera_exhausted",
            "available_image_count": len(render_evidence),
            "final_verdict": verdict,
            "final_confidence": confidence,
            "evidence_artifacts": list(render_evidence),
            "decision_source": (
                "nonrect_geometry_or_retained_visual_forced_binary"
            ),
        }
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


def _with_p0b_evidence_control(
    judge: Any,
    *,
    local_view_provider: LocalViewProvider | None,
) -> Any:
    """Keep raw experiment calls on the same provider-aware control boundary."""

    # A compatibility stub may opt out only by declaring that it is not a VLM.
    # Missing attributes, missing evidence, or a bare callable are never used to
    # infer an evidence-gate bypass.
    if getattr(judge, "vlm_control_enabled", None) is False:
        return judge
    from benchmark.visual_judge.control_config import (
        resolve_vlm_evaluation_control,
    )
    from benchmark.visual_judge.runtime import build_controlled_vlm_judge

    max_views = getattr(local_view_provider, "max_views", None)
    max_steps = getattr(local_view_provider, "max_steps", None)
    max_images = getattr(judge, "max_images", None)
    control = resolve_vlm_evaluation_control(
        existing_max_views=(
            max_views
            if isinstance(max_views, int)
            and not isinstance(max_views, bool)
            else None
        ),
        existing_max_steps=(
            max_steps
            if isinstance(max_steps, int)
            and not isinstance(max_steps, bool)
            else None
        ),
        existing_selector_available=local_view_provider is not None,
        judge_max_images=(
            max_images
            if isinstance(max_images, int)
            and not isinstance(max_images, bool)
            else None
        ),
    )
    return build_controlled_vlm_judge(
        judge,
        control=control,
        camera_provider=local_view_provider,
    )


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
    request = {
        "metric": metric_name,
        "event": _project_event_for_judge(metric_name, event),
        "scene": _project_scene_for_camera_evidence(scene),
        "object_ids": resolved_ids,
        "architecture_element": event.get("architecture_element"),
        "detector_evidence": _project_detector_evidence_for_judge(
            metric_name,
            detector_evidence,
        ),
        "access": "read_only_evidence_request",
    }
    if str(prompt or "").strip():
        request["natural_language_prompt"] = str(prompt)
        if relationships is not None:
            request["extracted_relationships"] = deepcopy(relationships)
    return request


def _project_scene_for_camera_evidence(
    scene: dict[str, Any],
) -> dict[str, Any]:
    """Preserve render geometry while withholding generator-private intent.

    Camera acquisition still receives the same object transforms and active
    wall geometry as the single-room baseline.  Task-slot intent and wall
    activation claims are not needed to choose or render a view and must not be
    recoverable through the compatibility camera request carried by the Judge.
    """

    projected = project_scene_for_evaluator_context(scene)
    metadata = projected.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        projected["metadata"] = metadata
    architecture = deepcopy(architecture_contract_from_scene(scene))
    physical = architecture.get("physical_walls")
    if isinstance(physical, dict):
        physical["policy_source"] = "withheld_from_evaluator"
        physical["activation_sources"] = []
        physical["activation_claims"] = []
    metadata["architecture_contract"] = architecture
    return projected


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


def _support_local_raw_count(
    *,
    local_paths: list[str],
    local_metadata: list[dict[str, Any]],
) -> int:
    if not local_metadata:
        # Path-only providers are the legacy local-view contract.
        return len(_deduplicate_paths(local_paths))
    return len(
        {
            str(item.get("view_id") or item.get("path") or "")
            for item in local_metadata
            if str(item.get("role") or "") in _SUPPORT_LOCAL_RAW_ROLES
            and str(item.get("view_id") or item.get("path") or "")
        }
    )


def _metric_local_view_count(
    *,
    metric: str,
    local_paths: list[str],
    local_metadata: list[dict[str, Any]],
) -> int:
    if not local_metadata:
        return len(_deduplicate_paths(local_paths))
    metric_name = str(metric).strip().lower()
    if metric_name == "collision":
        roles = _SUPPORT_LOCAL_RAW_ROLES | {
            "metric_local_contour",
            "collision_pair_overlay",
        }
    elif metric_name == "oob":
        roles = {
            "metric_local_rgb",
            "metric_local_highlight",
            "collision_rgb",
            "collision_pair_overlay",
        }
    else:
        roles = _SUPPORT_LOCAL_RAW_ROLES | {
            "metric_local_highlight",
        }
    return len(
        {
            str(item.get("view_id") or item.get("path") or "")
            for item in local_metadata
            if str(item.get("role") or "") in roles
            and str(item.get("view_id") or item.get("path") or "")
        }
    )


def _is_nonrect_scene(scene: dict[str, Any]) -> bool:
    metadata = scene.get("metadata")
    return bool(
        isinstance(metadata, dict)
        and metadata.get("evaluation_mode")
        == "non_rectangular_multi_room"
        and isinstance(
            metadata.get("non_rectangular_room_geometry"),
            dict,
        )
    )


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


def _project_event_for_judge(metric: str, event: dict) -> dict[str, Any]:
    """Keep the historical event while removing duplicated wall routing hints."""

    projected = deepcopy(event)
    if metric == "support":
        projected.pop("active_physical_wall_ids", None)
        projected.pop("architecture_contact_candidates", None)
    return projected


def _project_architecture_for_judge(
    metric: str,
    architecture: dict[str, Any],
) -> dict[str, Any]:
    """Expose baseline room geometry plus the narrow metric-owned wall delta.

    Activation provenance, policy claims, compatibility flags and allowed-token
    registries are benchmark plumbing rather than observable evidence. OOB and
    Support receive active physical-wall geometry without the source that
    activated those walls; Collision remains object-object only.
    """

    logical = architecture.get("logical_boundary")
    floor = architecture.get("floor")
    ceiling = architecture.get("ceiling")
    projected: dict[str, Any] = {
        "logical_boundary": {
            "enabled": bool(
                logical.get("enabled", True)
                if isinstance(logical, dict)
                else True
            ),
            "boundary": deepcopy(
                logical.get("boundary")
                if isinstance(logical, dict)
                else None
            ),
        },
        "floor": {
            "enabled": bool(
                floor.get("enabled", True)
                if isinstance(floor, dict)
                else True
            ),
            "z": (
                floor.get("z")
                if isinstance(floor, dict)
                else architecture.get("floor_z")
            ),
        },
        "ceiling": {
            "enabled": bool(
                ceiling.get("enabled", True)
                if isinstance(ceiling, dict)
                else True
            ),
            "z": ceiling.get("z") if isinstance(ceiling, dict) else None,
        },
    }
    if architecture.get("geometry_type") == "non_rectangular_polygon":
        projected["geometry_type"] = "non_rectangular_polygon"
        projected["ceiling"] = {"enabled": False, "z": None}
    if metric not in _PHYSICAL_WALL_METRICS:
        return projected
    physical = architecture.get("physical_walls")
    if not isinstance(physical, dict):
        return projected
    active_wall_ids = [
        str(value)
        for value in physical.get("active_wall_ids", [])
        if str(value)
    ]
    if not active_wall_ids:
        return projected
    thickness = physical.get("wall_thickness_m")
    projected["physical_walls"] = {
        "active_wall_ids": active_wall_ids,
        "wall_thickness_m": thickness,
        "center_plane": "logical_room_boundary",
        "inner_surface_offset_m": (
            float(thickness) / 2.0
            if isinstance(thickness, (int, float))
            and not isinstance(thickness, bool)
            else None
        ),
    }
    if architecture.get("geometry_type") == "non_rectangular_polygon":
        projected["physical_walls"]["wall_segments"] = deepcopy(
            physical.get("wall_segments") or []
        )
    return projected


def _project_detector_evidence_for_judge(
    metric: str,
    detector_evidence: dict[str, Any],
) -> dict[str, Any]:
    """Drop duplicated policy/provenance fields, preserving measured evidence."""

    projected = deepcopy(detector_evidence)
    projected.pop("extracted_relationships_are_claims_only", None)
    projected.pop("candidate_selection_policy", None)
    if metric != "support":
        return projected

    for key in (
        "support_instruction",
        "evaluated_object",
        "routing_reasons",
        "active_physical_wall_ids",
        "contact_fraction_affects_route",
        "center_ray_affects_route",
        "legacy_contact_tolerance_affects_direct_valid",
    ):
        projected.pop(key, None)

    wall_model = projected.get("architecture_wall_surface_model")
    if isinstance(wall_model, dict):
        projected["architecture_wall_surface_model"] = {
            key: deepcopy(wall_model.get(key))
            for key in (
                "active_wall_ids",
                "physical_wall_center_plane",
                "physical_wall_thickness_m",
                "inner_surface_offset_m",
            )
        }
    candidates = projected.get("architecture_contact_candidates")
    if isinstance(candidates, list):
        projected["architecture_contact_candidates"] = [
            {
                "plane": item.get("plane"),
                "signed_clearance_m": item.get("signed_clearance_m"),
                **(
                    {
                        "wall_id": item.get("wall_id"),
                        "distance_m": item.get("distance_m"),
                        "inward_normal_xy": deepcopy(
                            item.get("inward_normal_xy")
                        ),
                    }
                    if item.get("wall_id") is not None
                    else {}
                ),
            }
            for item in candidates
            if isinstance(item, dict)
        ]
    return projected


def _observable_architecture_for_scene(
    scene: dict[str, Any],
) -> dict[str, Any]:
    """Return observable architecture without generator activation claims."""

    from benchmark.non_rectangular.geometry import polygon_geometry_from_scene

    geometry = polygon_geometry_from_scene(scene)
    if geometry is None:
        return deepcopy(architecture_contract_from_scene(scene))
    thicknesses = [wall.thickness_m for wall in geometry.walls]
    return {
        "geometry_type": "non_rectangular_polygon",
        "logical_boundary": {
            "enabled": True,
            "boundary": [list(point) for point in geometry.floor_polygon_xy],
        },
        "floor": {"enabled": True, "z": geometry.floor_z_m},
        "ceiling": {"enabled": False, "z": None},
        "physical_walls": {
            "active_wall_ids": [wall.wall_id for wall in geometry.walls],
            "wall_thickness_m": (
                max(thicknesses) if thicknesses else None
            ),
            "wall_segments": [
                wall.public_dict() for wall in geometry.walls
            ],
        },
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
