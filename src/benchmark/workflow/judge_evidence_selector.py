from __future__ import annotations

import math
from typing import Any


DEFAULT_EVIDENCE_BUDGET = {
    "enabled": False,
    "judge_evidence_budgeting": False,
    "max_input_tokens": 60000,
    "include_global_view": True,
    "base_max_groups_for_judge": 3,
    "budget_raise_ratio": 0.5,
    "max_groups_for_judge_cap": 5,
    "group_views": ["xy", "yz", "xz"],
    "include_full_object_list": False,
    "include_compact_scene_summary": True,
    "include_selected_group_details": True,
    "summarize_schema_flags": True,
    "summarize_physical_flags": True,
    "summarize_view_flags": True,
}

PHYSICAL_FLAG_SELECTION_WEIGHTS = {
    "serious_collision": 80,
    "room_boundary": 70,
    "below_floor": 60,
    "above_wall_height": 60,
}
DEFAULT_PHYSICAL_FLAG_SELECTION_WEIGHT = 50


def resolve_vlm_judge_profile_config(benchmark_config: dict | None, runtime_profile: str | None) -> dict:
    vlm_judge = (benchmark_config or {}).get("vlm_judge")
    if not isinstance(vlm_judge, dict):
        return {"judge_generation": {}}

    default = vlm_judge.get("default") if isinstance(vlm_judge.get("default"), dict) else {}
    profiles = vlm_judge.get("profiles") if isinstance(vlm_judge.get("profiles"), dict) else {}
    profile = profiles.get(runtime_profile) if runtime_profile and isinstance(profiles.get(runtime_profile), dict) else {}
    return _merge_profile(default, profile)


def evidence_budgeting_config(
    benchmark_config: dict | None,
    judge_evidence_budgeting: bool | str | None = False,
    runtime_profile: str | None = None,
) -> dict:
    if isinstance(judge_evidence_budgeting, str) and runtime_profile is None:
        runtime_profile = judge_evidence_budgeting
        judge_evidence_budgeting = False
    config = _evidence_budget_section(benchmark_config)
    merged = dict(DEFAULT_EVIDENCE_BUDGET)
    if isinstance(config, dict):
        merged.update(config)
    enabled = bool(judge_evidence_budgeting)
    merged["enabled"] = enabled
    merged["judge_evidence_budgeting"] = enabled
    if runtime_profile is not None:
        merged["runtime_profile"] = runtime_profile
    return merged


def judge_generation_overrides(benchmark_config: dict | None, runtime_profile: str | None) -> dict:
    config = resolve_vlm_judge_profile_config(benchmark_config, runtime_profile).get("judge_generation", {})
    return dict(config) if isinstance(config, dict) else {}


def _evidence_budget_section(benchmark_config: dict | None) -> dict:
    vlm_judge = (benchmark_config or {}).get("vlm_judge")
    if not isinstance(vlm_judge, dict):
        return {}
    budget = vlm_judge.get("evidence_budget")
    return budget if isinstance(budget, dict) else {}


def select_judge_evidence(
    *,
    global_view_artifacts: list[dict],
    group_view_artifacts: list[dict],
    object_groups: list[dict],
    physical_flags: list[dict],
    view_flags: list[dict],
    render_skipped_objects: list[dict],
    config: dict,
    runtime_profile: str | None = None,
) -> dict:
    """Select a deterministic subset of rendered evidence for one VLM judge call."""

    budget = dict(DEFAULT_EVIDENCE_BUDGET)
    budget.update(config or {})
    if not bool(budget.get("enabled")):
        return _disabled_selection(global_view_artifacts, group_view_artifacts, object_groups, runtime_profile, budget)

    group_views = [str(item) for item in budget.get("group_views", ["xy", "yz", "xz"])]
    global_sent = _selected_global_views(global_view_artifacts, budget)
    limits = _budget_limits(budget, global_count=len(global_sent), group_view_count=len(group_views), total_groups=len(object_groups))
    max_images = limits["effective_max_images"]
    max_groups = limits["effective_max_groups_for_judge"]

    artifacts_by_group = _group_artifacts_by_group(group_view_artifacts)
    scored = [
        _score_group(group, physical_flags, view_flags, render_skipped_objects)
        for group in object_groups
        if isinstance(group, dict)
    ]
    scored.sort(key=lambda item: (-item["selection_score"], str(item["group_id"])))
    selected_ids = {item["group_id"] for item in scored[: max(0, max_groups)]}
    selected_groups = []
    omitted_groups = []
    selected_group_artifacts = []
    annotations = {}

    for item in scored:
        group_id = str(item["group_id"])
        object_ids = list(item.get("object_ids", []))
        annotation = {
            "sent_to_judge": group_id in selected_ids,
            "selection_score": int(item["selection_score"]),
            "selection_reasons": list(item["selection_reasons"]),
        }
        annotations[group_id] = annotation
        if group_id in selected_ids:
            views_sent = {}
            for projection in group_views:
                artifact = artifacts_by_group.get(group_id, {}).get(projection)
                if artifact:
                    views_sent[projection] = artifact.get("path")
                    selected_group_artifacts.append(artifact)
            selected_groups.append(
                {
                    "group_id": group_id,
                    "selection_score": int(item["selection_score"]),
                    "selection_reasons": list(item["selection_reasons"]),
                    "object_ids": object_ids,
                    "views_sent": views_sent,
                }
            )
        else:
            omitted_groups.append(
                {
                    "group_id": group_id,
                    "object_ids": object_ids,
                    "reason": "budget_limit",
                    "selection_score": int(item["selection_score"]),
                }
            )

    return {
        "budgeting_enabled": True,
        "judge_evidence_budgeting": True,
        "mode": "budgeted",
        "runtime_profile": runtime_profile,
        "budget": {
            "max_images": max_images,
            "base_max_groups_for_judge": limits["base_max_groups_for_judge"],
            "budget_raise_ratio": limits["budget_raise_ratio"],
            "effective_max_groups_for_judge": max_groups,
            "max_groups_for_judge_cap": limits["max_groups_for_judge_cap"],
            "effective_max_images": max_images,
            "selected_images": len(global_sent) + len(selected_group_artifacts),
            "max_groups_for_judge": max_groups,
            "max_input_tokens": budget.get("max_input_tokens"),
            "group_views": group_views,
        },
        "global_views_sent": [artifact.get("path") for artifact in global_sent],
        "selected_groups": selected_groups,
        "omitted_groups": omitted_groups,
        "selected_global_artifacts": global_sent,
        "selected_group_artifacts": selected_group_artifacts,
        "group_annotations": annotations,
        "budget_config": budget,
    }


def annotate_object_groups(object_groups: list[dict], selection: dict | None) -> list[dict]:
    if not isinstance(selection, dict) or not selection.get("budgeting_enabled"):
        return object_groups
    annotations = selection.get("group_annotations", {}) if isinstance(selection, dict) else {}
    annotated = []
    for group in object_groups:
        if not isinstance(group, dict):
            continue
        item = dict(group)
        annotation = annotations.get(group.get("group_id"), {})
        item.update(
            {
                "sent_to_judge": bool(annotation.get("sent_to_judge", False)),
                "selection_score": int(annotation.get("selection_score", 0)),
                "selection_reasons": list(annotation.get("selection_reasons", [])),
            }
        )
        annotated.append(item)
    return annotated


def selected_group_ids(selection: dict | None) -> set[str]:
    if not isinstance(selection, dict):
        return set()
    return {
        str(item.get("group_id"))
        for item in selection.get("selected_groups", [])
        if isinstance(item, dict) and item.get("group_id")
    }


def _merge_profile(default: dict, profile: dict) -> dict:
    merged = {
        "judge_generation": {},
    }
    for source in [default, profile]:
        if not isinstance(source, dict):
            continue
        for key, value in source.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                section = dict(merged[key])
                section.update(value)
                merged[key] = section
            else:
                merged[key] = value
    return merged


def _disabled_selection(global_view_artifacts: list[dict], group_view_artifacts: list[dict], object_groups: list[dict], runtime_profile: str | None, budget: dict) -> dict:
    sent_group_artifacts = [item for item in group_view_artifacts if item.get("id") != "camera_policy"]
    return {
        "budgeting_enabled": False,
        "judge_evidence_budgeting": False,
        "mode": "full",
        "runtime_profile": runtime_profile,
        "budget": {
            "max_images": budget.get("max_images"),
            "base_max_groups_for_judge": budget.get("base_max_groups_for_judge"),
            "budget_raise_ratio": budget.get("budget_raise_ratio"),
            "effective_max_groups_for_judge": None,
            "max_groups_for_judge_cap": budget.get("max_groups_for_judge_cap"),
            "effective_max_images": None,
            "selected_images": len([item for item in global_view_artifacts + group_view_artifacts if item.get("id") != "camera_policy"]),
            "max_groups_for_judge": budget.get("max_groups_for_judge"),
            "max_input_tokens": budget.get("max_input_tokens"),
            "group_views": budget.get("group_views", ["xy", "yz", "xz"]),
        },
        "global_views_sent": [artifact.get("path") for artifact in global_view_artifacts if artifact.get("id") != "camera_policy"],
        "selected_groups": [],
        "omitted_groups": [],
        "selected_global_artifacts": list(global_view_artifacts),
        "selected_group_artifacts": list(group_view_artifacts),
        "full_groups_sent": _full_group_records(object_groups, sent_group_artifacts),
        "group_annotations": {},
        "budget_config": budget,
    }


def _full_group_records(object_groups: list[dict], group_view_artifacts: list[dict]) -> list[dict]:
    artifacts_by_group = _group_artifacts_by_group(group_view_artifacts)
    records = []
    for group in object_groups:
        if not isinstance(group, dict):
            continue
        group_id = str(group.get("group_id"))
        records.append(
            {
                "group_id": group_id,
                "object_ids": list(group.get("object_ids", [])) if isinstance(group.get("object_ids"), list) else [],
                "views_sent": {
                    projection: artifact.get("path")
                    for projection, artifact in artifacts_by_group.get(group_id, {}).items()
                },
            }
        )
    return records


def _score_group(group: dict, physical_flags: list[dict], view_flags: list[dict], render_skipped_objects: list[dict]) -> dict:
    object_ids = list(group.get("object_ids", [])) if isinstance(group.get("object_ids"), list) else []
    object_set = {str(item) for item in object_ids}
    reasons: list[str] = []
    score = 0

    if _has_object_flag(render_skipped_objects, object_set):
        score += 100
        reasons.append("render_skipped_object")
    physical_score, physical_reasons = _physical_flag_selection_score(physical_flags, object_set)
    score += physical_score
    reasons.extend(physical_reasons)
    if _has_group_edge(group, {"support_parent", "attachment"}):
        score += 40
        reasons.append("support_or_attachment")
    if _has_group_edge(group, {"explicit_relation"}):
        score += 30
        reasons.append("explicit_relation")
    if group.get("group_source") == "semantic_region":
        score += 25
        reasons.append("semantic_region")
    if _has_view_flag(view_flags, str(group.get("group_id")), object_set):
        score += 20
        reasons.append("view_flag")

    score += 5 * len(object_ids)
    if object_ids:
        reasons.append("object_count")
    return {
        "group_id": group.get("group_id"),
        "object_ids": object_ids,
        "selection_score": score,
        "selection_reasons": reasons,
    }


def _has_object_flag(flags: list[dict], object_set: set[str]) -> bool:
    for flag in flags:
        objects = {str(item) for item in flag.get("objects", []) if item is not None} if isinstance(flag, dict) else set()
        object_id = flag.get("object_id") if isinstance(flag, dict) else None
        if object_id:
            objects.add(str(object_id))
        if objects & object_set:
            return True
    return False


def _physical_flag_selection_score(flags: list[dict], object_set: set[str]) -> tuple[int, list[str]]:
    weights_by_type: dict[str, int] = {}
    for flag in flags:
        if not isinstance(flag, dict):
            continue
        objects = {str(item) for item in flag.get("objects", []) if item is not None}
        if not (objects & object_set):
            continue
        flag_type = str(flag.get("type") or "physical_flag")
        weight = int(PHYSICAL_FLAG_SELECTION_WEIGHTS.get(flag_type, _severity_selection_weight(flag.get("severity"))))
        weights_by_type[flag_type] = max(weights_by_type.get(flag_type, 0), weight)
    if not weights_by_type:
        return 0, []
    return min(140, sum(weights_by_type.values())), sorted(weights_by_type, key=lambda item: (-weights_by_type[item], item))


def _severity_selection_weight(severity: object) -> int:
    text = str(severity or "").lower()
    if text in {"critical", "high"}:
        return 70
    if text in {"major", "medium"}:
        return 60
    if text in {"minor", "low", "warning"}:
        return 40
    return DEFAULT_PHYSICAL_FLAG_SELECTION_WEIGHT


def _has_group_edge(group: dict, reasons: set[str]) -> bool:
    edge_reasons = {str(item) for item in group.get("edge_reasons", []) if item is not None}
    if edge_reasons & reasons:
        return True
    for edge in group.get("formation_edges", []):
        if isinstance(edge, dict) and str(edge.get("reason")) in reasons:
            return True
    return False


def _has_view_flag(flags: list[dict], group_id: str, object_set: set[str]) -> bool:
    for flag in flags:
        if not isinstance(flag, dict):
            continue
        if str(flag.get("group_id")) == group_id:
            return True
        objects = {str(item) for item in flag.get("objects", []) if item is not None}
        if objects & object_set:
            return True
    return False


def _selected_global_views(global_view_artifacts: list[dict], budget: dict) -> list[dict]:
    if not bool(budget.get("include_global_view", True)):
        return []
    for artifact in global_view_artifacts:
        if artifact.get("id") != "camera_policy":
            return [artifact]
    return []


def _group_artifacts_by_group(group_view_artifacts: list[dict]) -> dict[str, dict[str, dict]]:
    grouped: dict[str, dict[str, dict]] = {}
    for artifact in group_view_artifacts:
        if not isinstance(artifact, dict) or artifact.get("id") == "camera_policy":
            continue
        group_id, projection = _group_and_projection(artifact)
        if group_id and projection:
            grouped.setdefault(group_id, {})[projection] = artifact
    return grouped


def _group_and_projection(artifact: dict) -> tuple[str, str]:
    artifact_id = str(artifact.get("id") or "")
    for projection in ["xy", "yz", "xz"]:
        suffix = f"_{projection}"
        if artifact_id.endswith(suffix):
            return artifact_id[: -len(suffix)], projection
    path = str(artifact.get("path") or "")
    parts = path.replace("\\", "/").split("/")
    group_id = next((part for part in parts if part.startswith("group_")), "")
    return group_id, ""


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _budget_limits(budget: dict, *, global_count: int, group_view_count: int, total_groups: int) -> dict:
    base_groups = _optional_int(budget.get("base_max_groups_for_judge"))
    if base_groups is None:
        base_groups = _optional_int(budget.get("max_groups_for_judge"))
    if base_groups is None:
        base_groups = total_groups
    raise_ratio = _optional_float(budget.get("budget_raise_ratio"), 0.0)
    cap = _optional_int(budget.get("max_groups_for_judge_cap"))
    effective_groups = int(math.ceil(max(0, base_groups) * (1.0 + max(0.0, raise_ratio))))
    if cap is not None:
        effective_groups = min(effective_groups, max(0, cap))
    effective_images = global_count + effective_groups * max(0, group_view_count)
    return {
        "base_max_groups_for_judge": base_groups,
        "budget_raise_ratio": raise_ratio,
        "max_groups_for_judge_cap": cap,
        "effective_max_groups_for_judge": effective_groups,
        "effective_max_images": effective_images,
    }


def _optional_float(value: Any, default: float) -> float:
    if value is None or value == "":
        return default
    return float(value)
