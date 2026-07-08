from __future__ import annotations

from collections import defaultdict

from benchmark.evidence_config import effective_scale_aware_min_volume, scene_volume_m3_from_case


DEFAULT_COLLISION_AVOIDANCE_CONFIG = {
    "enabled": True,
    "soft_overlap_ratio": 0.15,
    "soft_min_volume": {
        "abs_min_volume_m3": 0.001,
        "object_volume_ratio": 0.005,
        "scene_volume_ratio": 0.00005,
        "min_cap_m3": 0.001,
        "max_cap_m3": 0.03,
    },
    "max_pairs": 12,
    "cost_mode": "normalized_dimensionless",
    "weights": {
        "outside": 1.0,
        "overlap": 1.0,
        "movement": 0.25,
    },
}

DEFAULT_COLLISION_REPAIR_CONFIG = {
    "dense_cluster_min_objects": 4,
    "dense_cluster_min_edges": 6,
    "dense_cluster_max_clusters": 5,
    "max_pair_cues_per_cluster": 8,
    "aggregate_min_collision_count": 2,
    "aggregate_max_magnitude_m": 1.5,
}


def build_feedback(
    evaluation_report: dict,
    current_layout: dict,
    bm_instance: dict,
    benchmark_config: dict | None = None,
) -> dict:
    collision_cfg = _collision_avoidance_config(benchmark_config)
    collision_repair_cfg = _collision_repair_config(benchmark_config)
    soft_collision_pairs = _soft_collision_pairs(
        current_layout,
        collision_cfg,
        bm_instance=bm_instance,
        serious_pairs=_debug_serious_collision_pairs(evaluation_report),
    )
    repair_targets = sorted(
        set(evaluation_report.get("repair_targets", []))
        | _debug_physical_repair_targets(evaluation_report)
        | _soft_collision_targets(soft_collision_pairs)
    )
    repair_target_set = set(repair_targets)
    object_ids = sorted(
        obj.get("object_id")
        for obj in current_layout.get("objects", [])
        if isinstance(obj, dict) and isinstance(obj.get("object_id"), str)
    )
    locked_objects = [object_id for object_id in object_ids if object_id not in repair_target_set]

    violations = []
    violations.extend(_failures_to_violations("schema", evaluation_report.get("schema_failures", [])))
    violations.extend(_failures_to_violations("physical", evaluation_report.get("physical_failures", [])))
    violations.extend(_failures_to_violations("spatial_relation", evaluation_report.get("spatial_relation_failures", [])))
    debug_evidence = evaluation_report.get("debug_evidence", {})
    physical_flags = debug_evidence.get("physical_flags", []) if isinstance(debug_evidence, dict) else []
    vlm_judgement = evaluation_report.get("vlm_judgement", {})
    vlm_issues = vlm_judgement.get("issues", []) if isinstance(vlm_judgement, dict) else []
    violations.extend(_debug_flags_to_violations("physical_debug_flag", physical_flags))
    violations.extend(_vlm_issues_to_violations(vlm_issues))
    room_consistency = evaluation_report.get("room_consistency", {})
    if room_consistency.get("short_reason") and room_consistency.get("score", 4) <= 2:
        violations.append(
            {
                "category": "room_consistency",
                "type": "vlm_room_judge",
                "message": room_consistency["short_reason"],
                "objects": [],
            }
        )

    repair_actions = _repair_actions(
        repair_targets=repair_targets,
        current_layout=current_layout,
        bm_instance=bm_instance,
        physical_flags=physical_flags,
        vlm_issues=vlm_issues,
        soft_collision_pairs=soft_collision_pairs,
        collision_cfg=collision_cfg,
        collision_repair_cfg=collision_repair_cfg,
    )
    suggested_actions = _suggested_actions(repair_actions)
    issues = _general_issues(violations, evaluation_report)
    return {
        "scene_id": evaluation_report.get("scene_id")
        or evaluation_report.get("case_id")
        or evaluation_report.get("task_id")
        or bm_instance.get("scene_id")
        or bm_instance.get("case_id")
        or bm_instance.get("task_id", "unknown_scene"),
        "task_id": evaluation_report.get("task_id", bm_instance.get("task_id", "unknown_task")),
        "iteration": int(evaluation_report.get("iteration", 0)),
        "overall_valid": bool(evaluation_report.get("overall_valid", False)),
        "score": _feedback_score(evaluation_report, vlm_judgement),
        "score_norm": _feedback_score_norm(evaluation_report, vlm_judgement),
        "issues": issues,
        "repair_hints": _repair_hints(suggested_actions, issues),
        "physical_evidence": _physical_evidence(evaluation_report, debug_evidence),
        "vlm_judge_feedback": _vlm_judge_feedback(vlm_judgement, evaluation_report),
        "suggested_actions": suggested_actions,
        "advisory": True,
        "repair_targets": repair_targets,
        "locked_objects": locked_objects,
        "violations": violations,
        "repair_actions": repair_actions,
        "debug_evidence_summary": _debug_evidence_summary(debug_evidence),
        "room_consistency_reason": room_consistency.get("short_reason", ""),
        "instruction": (
            "Review the listed issues and suggested actions as advisory evaluation feedback. "
            "Use them only when repairing or improving the scene."
        ),
    }


def _general_issues(violations: list[dict], evaluation_report: dict) -> list[dict]:
    issues = []
    for violation in violations:
        if not isinstance(violation, dict):
            continue
        objects = violation.get("objects") or violation.get("object_ids") or []
        issues.append(
            {
                "source": violation.get("category", "evaluation"),
                "type": violation.get("type", "unknown"),
                "severity": violation.get("severity", "medium"),
                "object_ids": [item for item in objects if isinstance(item, str)],
                "message": violation.get("message", ""),
                "repair_hint": violation.get("repair_hint", ""),
            }
        )
    for failure in evaluation_report.get("hard_failures", []):
        if not isinstance(failure, dict):
            continue
        issues.append(
            {
                "source": failure.get("source", "evaluation"),
                "type": failure.get("code", "hard_failure"),
                "severity": "critical",
                "object_ids": [],
                "message": failure.get("message", ""),
                "repair_hint": "",
            }
        )
    return issues


def _feedback_score(evaluation_report: dict, vlm_judgement: object) -> object:
    if isinstance(vlm_judgement, dict) and "score" in vlm_judgement:
        return vlm_judgement.get("score")
    room = evaluation_report.get("room_consistency")
    if isinstance(room, dict) and "score" in room:
        return room.get("score")
    metrics = evaluation_report.get("metrics")
    return metrics.get("primary_score") if isinstance(metrics, dict) else None


def _feedback_score_norm(evaluation_report: dict, vlm_judgement: object) -> object:
    if isinstance(vlm_judgement, dict) and "score_norm" in vlm_judgement:
        return vlm_judgement.get("score_norm")
    room = evaluation_report.get("room_consistency")
    if isinstance(room, dict) and "score_norm" in room:
        return room.get("score_norm")
    return None


def _physical_evidence(evaluation_report: dict, debug_evidence: object) -> dict:
    debug = debug_evidence if isinstance(debug_evidence, dict) else {}
    deterministic = evaluation_report.get("deterministic_evidence")
    deterministic_physical = deterministic.get("physical_flags") if isinstance(deterministic, dict) else []
    physical_flags = debug.get("physical_flags", deterministic_physical)
    geometry_missing = debug.get("geometry_missing_assets")
    if geometry_missing is None and isinstance(debug.get("legend_compat"), dict):
        geometry_missing = debug["legend_compat"].get("bbox_missing_assets")
    return {
        "physical_flags": _compact_flags(physical_flags, limit=80),
        "geometry_missing_assets": _compact_flags(geometry_missing, limit=80),
        "render_skipped_objects": _compact_flags(debug.get("render_skipped_objects"), limit=80),
        "geometry_available_rate": evaluation_report.get("geometry_available_rate"),
        "render_evidence_used": bool(evaluation_report.get("render_evidence_used", False)),
        "json_scene_used": bool(evaluation_report.get("json_scene_used", False)),
    }


def _vlm_judge_feedback(vlm_judgement: object, evaluation_report: dict) -> dict:
    judgement = vlm_judgement if isinstance(vlm_judgement, dict) else {}
    return {
        "valid": judgement.get("valid"),
        "score": judgement.get("score"),
        "score_norm": judgement.get("score_norm"),
        "confidence": judgement.get("confidence"),
        "judgement_status": judgement.get("judgement_status") or evaluation_report.get("judgement_status"),
        "brief_reasoning": judgement.get("brief_reasoning") or judgement.get("short_reason", ""),
        "issues": judgement.get("issues", []) if isinstance(judgement.get("issues"), list) else [],
        "insufficient_evidence": bool(judgement.get("insufficient_evidence", False)),
    }


def _suggested_actions(repair_actions: list[dict]) -> list[dict]:
    actions = []
    for action in repair_actions:
        if not isinstance(action, dict):
            continue
        item = dict(action)
        item.setdefault("advisory", True)
        actions.append(item)
    return actions


def _repair_hints(suggested_actions: list[dict], issues: list[dict]) -> list[dict]:
    hints = []
    seen = set()
    for issue in issues:
        hint = issue.get("repair_hint") if isinstance(issue, dict) else None
        if isinstance(hint, str) and hint.strip():
            key = ("issue", hint.strip())
            if key not in seen:
                hints.append(
                    {
                        "source": issue.get("source", "evaluation"),
                        "object_ids": issue.get("object_ids", []),
                        "hint": hint.strip(),
                        "advisory": True,
                    }
                )
                seen.add(key)
    for action in suggested_actions:
        hint = action.get("repair_hint") or action.get("reason") or action.get("reason_code") or action.get("action")
        if not isinstance(hint, str) or not hint.strip():
            continue
        object_ids = action.get("object_ids")
        if not isinstance(object_ids, list):
            object_ids = [action.get("object_id")] if isinstance(action.get("object_id"), str) else []
        key = ("action", action.get("action"), tuple(object_ids), hint.strip())
        if key in seen:
            continue
        hints.append(
            {
                "source": action.get("action", "suggested_action"),
                "object_ids": [item for item in object_ids if isinstance(item, str)],
                "hint": hint.strip(),
                "advisory": True,
            }
        )
        seen.add(key)
    return hints


def _collision_avoidance_config(benchmark_config: dict | None) -> dict:
    repair = benchmark_config.get("repair", {}) if isinstance(benchmark_config, dict) else {}
    override = repair.get("collision_avoidance", {}) if isinstance(repair, dict) else {}
    if not isinstance(override, dict):
        override = {}
    return _merge_collision_config(DEFAULT_COLLISION_AVOIDANCE_CONFIG, override)


def _collision_repair_config(benchmark_config: dict | None) -> dict:
    config = benchmark_config if isinstance(benchmark_config, dict) else {}
    repair = config.get("repair", {}) if isinstance(config.get("repair"), dict) else {}
    override = repair.get("collision_repair") if isinstance(repair.get("collision_repair"), dict) else config.get("collision_repair", {})
    if not isinstance(override, dict):
        override = {}
    return _merge_collision_config(DEFAULT_COLLISION_REPAIR_CONFIG, override)


def _merge_collision_config(defaults: dict, overrides: dict) -> dict:
    merged = dict(defaults)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            nested = dict(merged[key])
            nested.update(value)
            merged[key] = nested
        else:
            merged[key] = value
    return merged


def _failures_to_violations(category: str, failures: list[dict]) -> list[dict]:
    violations = []
    for failure in failures:
        violation = {
            "category": category,
            "type": failure.get("type", "unknown"),
            "message": failure.get("message", ""),
        }
        if "objects" in failure:
            violation["objects"] = list(failure["objects"])
        if "category" in failure and category == "spatial_relation":
            violation["target_category"] = failure["category"]
        if "hard" in failure:
            violation["hard"] = bool(failure["hard"])
        violations.append(violation)
    return violations


def _debug_physical_repair_targets(evaluation_report: dict) -> set[str]:
    debug = evaluation_report.get("debug_evidence")
    if not isinstance(debug, dict):
        return set()
    targets = set()
    for flag in debug.get("physical_flags", []):
        if not isinstance(flag, dict):
            continue
        if flag.get("repair_relevant") is False:
            continue
        flag_type = flag.get("type")
        if flag_type not in {
            "room_boundary",
            "below_floor",
            "above_wall_height",
            "serious_collision",
            "floating_or_vertical_inconsistency",
        }:
            continue
        for object_id in flag.get("objects", []):
            if isinstance(object_id, str) and object_id:
                targets.add(object_id)
    return targets


def _debug_serious_collision_pairs(evaluation_report: dict) -> set[frozenset[str]]:
    debug = evaluation_report.get("debug_evidence")
    if not isinstance(debug, dict):
        return set()
    pairs: set[frozenset[str]] = set()
    for flag in debug.get("physical_flags", []):
        if not isinstance(flag, dict) or flag.get("type") != "serious_collision":
            continue
        object_ids = [item for item in flag.get("objects", []) if isinstance(item, str)]
        if len(object_ids) >= 2:
            pairs.add(frozenset(object_ids[:2]))
    return pairs


def _soft_collision_targets(soft_collision_pairs: list[dict]) -> set[str]:
    targets: set[str] = set()
    for pair in soft_collision_pairs:
        for object_id in pair.get("objects", []):
            if isinstance(object_id, str) and object_id:
                targets.add(object_id)
    return targets


def _debug_flags_to_violations(category: str, flags: object) -> list[dict]:
    if not isinstance(flags, list):
        return []
    violations = []
    for flag in flags[:80]:
        if not isinstance(flag, dict):
            continue
        if flag.get("repair_relevant") is False:
            continue
        flag_type = flag.get("type")
        if flag_type not in {
            "room_boundary",
            "below_floor",
            "above_wall_height",
            "serious_collision",
            "floating_or_vertical_inconsistency",
            "impossible_height_constraint",
        }:
            continue
        violations.append(
            {
                "category": category,
                "type": flag_type,
                "code": flag.get("code"),
                "message": flag.get("message", ""),
                "objects": [item for item in flag.get("objects", []) if isinstance(item, str)],
                "severity": flag.get("severity"),
                "confidence": flag.get("confidence"),
                "source_kind": flag.get("source_kind"),
                "source_confidence": flag.get("source_confidence"),
                "blocking": bool(flag.get("blocking", False)),
            }
        )
    if len(flags) > 80:
        violations.append(
            {
                "category": category,
                "type": "truncated",
                "message": f"{len(flags) - 80} additional debug physical flags omitted from repair prompt.",
                "objects": [],
            }
        )
    return violations


def _vlm_issues_to_violations(issues: object) -> list[dict]:
    if not isinstance(issues, list):
        return []
    violations = []
    for issue in issues[:20]:
        if not isinstance(issue, dict):
            continue
        object_ids = [item for item in issue.get("object_ids", []) if isinstance(item, str)]
        violations.append(
            {
                "category": "vlm_judge_issue",
                "type": issue.get("issue_type", "unknown"),
                "message": issue.get("evidence", ""),
                "objects": object_ids,
                "severity": issue.get("severity"),
                "repair_hint": issue.get("repair_hint", ""),
                "group_id": issue.get("group_id"),
            }
        )
    if len(issues) > 20:
        violations.append(
            {
                "category": "vlm_judge_issue",
                "type": "truncated",
                "message": f"{len(issues) - 20} additional VLM judge issues omitted from repair prompt.",
                "objects": [],
            }
        )
    return violations


def _repair_actions(
    *,
    repair_targets: list[str],
    current_layout: dict,
    bm_instance: dict,
    physical_flags: object,
    vlm_issues: object,
    soft_collision_pairs: list[dict],
    collision_cfg: dict,
    collision_repair_cfg: dict,
) -> list[dict]:
    objects = _layout_object_map(current_layout)
    regions = _floor_plan_regions(bm_instance)
    actions: list[dict] = []
    physical_flag_list = physical_flags if isinstance(physical_flags, list) else []
    impossible_height_objects = {
        object_id
        for flag in physical_flag_list
        if isinstance(flag, dict) and flag.get("type") == "impossible_height_constraint"
        for object_id in flag.get("objects", [])
        if isinstance(object_id, str)
    }
    actions.extend(_dense_collision_cluster_actions(physical_flag_list, objects, collision_repair_cfg))
    serious_pair_actions: list[dict] = []
    for flag in physical_flag_list:
        if not isinstance(flag, dict):
            continue
        if flag.get("repair_relevant") is False:
            continue
        flag_type = flag.get("type")
        object_ids = [item for item in flag.get("objects", []) if isinstance(item, str)]
        if flag_type == "room_boundary":
            for object_id in object_ids:
                obj = objects.get(object_id)
                if obj is None:
                    continue
                suggestion = _suggest_inside_floor_plan(obj, regions)
                actions.append(
                    {
                        "action": "move_inside_boundary",
                        "object_id": object_id,
                        "current_bbox": _object_bbox_summary(obj),
                        "target_region": suggestion.get("target_region"),
                        "suggested_center": suggestion.get("suggested_center"),
                        "suggested_delta": suggestion.get("suggested_delta"),
                        "suggested_delta_xy": suggestion.get("suggested_delta_xy"),
                        "distance_outside": suggestion.get("distance_outside"),
                        "boundary_source_kind": flag.get("source_kind") or suggestion.get("source_kind"),
                        "boundary_source_confidence": flag.get("source_confidence") or flag.get("confidence"),
                        "confidence": flag.get("confidence", "medium"),
                        "advisory": True,
                        "fallback_note": (
                            "Boundary is fallback-derived; treat this as a soft repair cue, not absolute room geometry."
                            if flag.get("confidence") == "low" or _is_fallback_source(flag.get("source_kind"))
                            else ""
                        ),
                        "reason": flag.get("message", "object footprint is outside the floor plan"),
                    }
                )
        elif flag_type == "below_floor":
            for object_id in object_ids:
                obj = objects.get(object_id)
                if obj is None:
                    continue
                center = _numeric_triplet(obj.get("center"))
                size = _numeric_triplet(obj.get("size"))
                if center is None or size is None:
                    continue
                actions.append(_height_repair_action(obj, bm_instance, flag, reason="below_floor"))
        elif flag_type == "above_wall_height":
            for object_id in object_ids:
                if object_id in impossible_height_objects:
                    continue
                obj = objects.get(object_id)
                if obj is None:
                    continue
                actions.append(_height_repair_action(obj, bm_instance, flag, reason="above_wall_height"))
        elif flag_type == "impossible_height_constraint":
            for object_id in object_ids:
                obj = objects.get(object_id)
                if obj is None:
                    continue
                actions.append(_impossible_height_action(obj, bm_instance, flag))
        elif flag_type == "floating_or_vertical_inconsistency":
            for object_id in object_ids:
                obj = objects.get(object_id)
                if obj is None:
                    continue
                actions.append(
                    {
                        "action": "lower_or_support_object",
                        "object_id": object_id,
                        "current_bbox": _object_bbox_summary(obj),
                        "bottom_z": flag.get("bottom_z"),
                        "nearest_support_top_z": flag.get("nearest_support_top_z"),
                        "vertical_gap": flag.get("vertical_gap"),
                        "confidence": flag.get("confidence", "medium"),
                        "advisory": True,
                        "reason": flag.get("message", "object appears vertically unsupported or floating"),
                    }
                )
        elif flag_type == "serious_collision" and len(object_ids) >= 2:
            obj_a = objects.get(object_ids[0])
            obj_b = objects.get(object_ids[1])
            if obj_a is None or obj_b is None:
                continue
            action = _separate_collision_action(obj_a, obj_b, flag, regions, objects, collision_cfg=collision_cfg)
            if action:
                actions.append(action)
                serious_pair_actions.append(action)

    actions.extend(_aggregate_collision_actions(serious_pair_actions, collision_repair_cfg))

    for pair in soft_collision_pairs:
        object_ids = [item for item in pair.get("objects", []) if isinstance(item, str)]
        if len(object_ids) < 2:
            continue
        obj_a = objects.get(object_ids[0])
        obj_b = objects.get(object_ids[1])
        if obj_a is None or obj_b is None:
            continue
        action = _separate_collision_action(
            obj_a,
            obj_b,
            {
                "type": "soft_collision",
                "objects": object_ids[:2],
                "overlap_ratio": pair.get("overlap_ratio"),
                "intersection_volume_m3": pair.get("intersection_volume_m3"),
                "smaller_object_volume_m3": pair.get("smaller_object_volume_m3"),
                "effective_soft_min_volume_m3": pair.get("effective_soft_min_volume_m3"),
                "threshold_source": pair.get("threshold_source"),
                "message": pair.get("message", "partial bbox overlap"),
            },
            regions,
            objects,
            collision_cfg=collision_cfg,
        )
        if action:
            action["soft_collision"] = True
            if "effective_soft_min_volume_m3" in pair:
                action["effective_soft_min_volume_m3"] = pair["effective_soft_min_volume_m3"]
            if "smaller_object_volume_m3" in pair:
                action["smaller_object_volume_m3"] = pair["smaller_object_volume_m3"]
            if "threshold_source" in pair:
                action["threshold_source"] = pair["threshold_source"]
            actions.append(action)

    # Keep the VLM's repair hints, but do not rely on them as the only repair signal.
    if isinstance(vlm_issues, list):
        for issue in vlm_issues[:10]:
            if not isinstance(issue, dict) or not issue.get("repair_hint"):
                continue
            actions.append(
                {
                    "action": "vlm_repair_hint",
                    "issue_type": issue.get("issue_type"),
                    "severity": issue.get("severity"),
                    "object_ids": [item for item in issue.get("object_ids", []) if isinstance(item, str)],
                    "repair_hint": issue.get("repair_hint"),
                }
            )

    if repair_targets and not actions:
        for object_id in repair_targets:
            obj = objects.get(object_id)
            if obj is None:
                continue
            actions.append(
                {
                    "action": "inspect_and_adjust_target",
                    "object_id": object_id,
                    "current_bbox": _object_bbox_summary(obj),
                    "reason": "target was selected for repair but no deterministic action was available",
                }
            )
    return actions


def _debug_evidence_summary(debug_evidence: object) -> dict:
    if not isinstance(debug_evidence, dict):
        return {}
    manifest = debug_evidence.get("judge_input_manifest")
    if not isinstance(manifest, dict):
        manifest = {}
    return {
        "sanity_flags": _compact_flags(debug_evidence.get("sanity_flags"), limit=20),
        "physical_flags": _compact_flags(debug_evidence.get("physical_flags"), limit=40),
        "view_flags": _compact_flags(debug_evidence.get("view_flags"), limit=20),
        "render_skipped_objects": _compact_flags(debug_evidence.get("render_skipped_objects"), limit=20),
        "selected_groups": _compact_groups(manifest.get("selected_groups"), limit=10),
        "omitted_groups": _compact_groups(manifest.get("omitted_groups"), limit=20),
    }


def _compact_flags(value: object, *, limit: int) -> list[dict]:
    if not isinstance(value, list):
        return []
    flags = []
    for item in value[:limit]:
        if not isinstance(item, dict):
            continue
        flags.append(
            {
                key: item[key]
                for key in [
                    "type",
                    "code",
                    "severity",
                    "confidence",
                    "source_kind",
                    "source_confidence",
                    "blocking",
                    "suppressed",
                    "repair_relevant",
                    "objects",
                    "object_ids",
                    "object_id",
                    "message",
                    "group_id",
                    "projection",
                    "view_id",
                    "overlap_ratio",
                    "threshold",
                    "vertical_gap",
                ]
                if key in item
            }
        )
    if len(value) > limit:
        flags.append({"type": "truncated", "message": f"{len(value) - limit} additional flags omitted."})
    return flags


def _compact_groups(value: object, *, limit: int) -> list[dict]:
    if not isinstance(value, list):
        return []
    groups = []
    for item in value[:limit]:
        if not isinstance(item, dict):
            continue
        groups.append(
            {
                key: item[key]
                for key in ["group_id", "object_ids", "reason", "selection_score", "selection_reasons"]
                if key in item
            }
        )
    if len(value) > limit:
        groups.append({"group_id": "truncated", "reason": f"{len(value) - limit} additional groups omitted."})
    return groups


def _soft_collision_pairs(
    layout: dict,
    collision_cfg: dict,
    *,
    bm_instance: dict,
    serious_pairs: set[frozenset[str]],
) -> list[dict]:
    if not bool(collision_cfg.get("enabled", True)):
        return []
    objects = _layout_object_map(layout)
    object_items = sorted(objects.items())
    soft_ratio = float(collision_cfg.get("soft_overlap_ratio", 0.15))
    min_volume_config = collision_cfg.get("soft_min_volume")
    scene_volume = scene_volume_m3_from_case(bm_instance)
    max_pairs = int(collision_cfg.get("max_pairs", 12))
    pairs = []
    for index, (id_a, obj_a) in enumerate(object_items):
        volume_a = _object_volume(obj_a)
        if volume_a <= 0:
            continue
        for id_b, obj_b in object_items[index + 1 :]:
            pair_key = frozenset([id_a, id_b])
            if pair_key in serious_pairs:
                continue
            center_b = _numeric_triplet(obj_b.get("center"))
            if center_b is None:
                continue
            volume_b = _object_volume(obj_b)
            if volume_b <= 0:
                continue
            overlap = _candidate_overlap_volume(obj_a, obj_b, center_b)
            smaller_volume = min(volume_a, volume_b)
            threshold = effective_scale_aware_min_volume(
                min_volume_config if isinstance(min_volume_config, dict) else {},
                smaller_object_volume_m3=smaller_volume,
                scene_volume_m3=scene_volume,
            )
            effective_min_volume = float(threshold["effective_min_collision_volume_m3"])
            if overlap <= effective_min_volume:
                continue
            ratio = overlap / max(1.0e-9, min(volume_a, volume_b))
            if ratio <= soft_ratio:
                continue
            pairs.append(
                {
                    "type": "soft_collision",
                    "objects": [id_a, id_b],
                    "overlap_ratio": round(ratio, 6),
                    "intersection_volume_m3": round(overlap, 6),
                    "smaller_object_volume_m3": round(smaller_volume, 6),
                    "effective_soft_min_volume_m3": round(effective_min_volume, 6),
                    "threshold_source": threshold["threshold_source"],
                    "threshold": soft_ratio,
                    "message": (
                        f"{id_a} partially overlaps {id_b} above soft collision ratio "
                        f"{soft_ratio:.3f}; reduce or eliminate this overlap during repair."
                    ),
                }
            )
    pairs.sort(key=lambda item: (-(item.get("overlap_ratio") or 0), item["objects"]))
    return pairs[:max_pairs]


def _layout_object_map(layout: dict) -> dict[str, dict]:
    return {
        obj["object_id"]: obj
        for obj in layout.get("objects", [])
        if isinstance(obj, dict) and isinstance(obj.get("object_id"), str)
    }


def _floor_plan_regions(bm_instance: dict) -> list[dict]:
    room = bm_instance.get("room") if isinstance(bm_instance.get("room"), dict) else {}
    floor_plan = room.get("floor_plan") if isinstance(room.get("floor_plan"), dict) else {}
    regions = floor_plan.get("regions") or room.get("regions") or []
    source_kind = floor_plan.get("source_kind") or room.get("boundary_source_kind") or room.get("source_kind")
    valid_regions = []
    for region in regions:
        if not isinstance(region, dict) or _region_bounds(region) is None:
            continue
        copied = dict(region)
        copied.setdefault("source_kind", source_kind or "semantic_region")
        valid_regions.append(copied)
    if valid_regions:
        return valid_regions

    for source_key, polygon in [
        ("floor_plan.aggregate_boundary", floor_plan.get("aggregate_boundary")),
        ("room.floor_polygon", room.get("floor_polygon")),
        ("room.boundary", room.get("boundary")),
    ]:
        region = _synthetic_floor_plan_region(polygon, source_key)
        if region is not None:
            return [region]
    return []


def _synthetic_floor_plan_region(polygon: object, source_key: str) -> dict | None:
    if not isinstance(polygon, list) or not polygon:
        return None
    region = {
        "id": "__aggregate_floor_plan__",
        "label": "aggregate_floor_plan",
        "floor_polygon": polygon,
        "source": source_key,
        "source_kind": "object_position_extent_fallback" if "boundary" in source_key or "aggregate" in source_key else "room_metadata",
        "synthetic": True,
    }
    return region if _region_bounds(region) is not None else None


def _suggest_inside_floor_plan(obj: dict, regions: list[dict]) -> dict:
    center = _numeric_triplet(obj.get("center"))
    size = _numeric_triplet(obj.get("size"))
    if center is None or size is None or not regions:
        return {
            "target_region": None,
            "suggested_center": center,
            "suggested_delta": [0.0, 0.0, 0.0],
            "suggested_delta_xy": [0.0, 0.0],
            "distance_outside": 0.0,
        }

    region = _matching_region(obj.get("region_id"), regions) or min(
        regions,
        key=lambda item: _distance_to_bounds(center[0], center[1], _region_bounds(item) or (0.0, 0.0, 0.0, 0.0)),
    )
    bounds = _region_bounds(region)
    if bounds is None:
        return {
            "target_region": None,
            "suggested_center": _round_list(center),
            "suggested_delta": [0.0, 0.0, 0.0],
            "suggested_delta_xy": [0.0, 0.0],
            "distance_outside": 0.0,
        }

    min_x, max_x, min_y, max_y = bounds
    margin = 0.02
    half_w = max(0.0, size[0] / 2.0)
    half_d = max(0.0, size[1] / 2.0)
    safe_min_x = min_x + half_w + margin
    safe_max_x = max_x - half_w - margin
    safe_min_y = min_y + half_d + margin
    safe_max_y = max_y - half_d - margin
    if safe_min_x > safe_max_x:
        safe_min_x = safe_max_x = (min_x + max_x) / 2.0
    if safe_min_y > safe_max_y:
        safe_min_y = safe_max_y = (min_y + max_y) / 2.0
    suggested = [
        _clamp(center[0], safe_min_x, safe_max_x),
        _clamp(center[1], safe_min_y, safe_max_y),
        max(center[2], size[2] / 2.0),
    ]
    delta = [suggested[i] - center[i] for i in range(3)]
    return {
        "target_region": _region_summary(region),
        "suggested_center": _round_list(suggested),
        "suggested_delta": _round_list(delta),
        "suggested_delta_xy": _round_list(delta[:2]),
        "distance_outside": round(_outside_floor_plan_penalty(obj, center, [region]), 4),
        "source_kind": region.get("source_kind"),
    }


def _separate_collision_action(
    obj_a: dict,
    obj_b: dict,
    flag: dict,
    regions: list[dict],
    all_objects: dict[str, dict],
    *,
    collision_cfg: dict,
) -> dict | None:
    candidates = _collision_separation_candidates(obj_a, obj_b, regions)
    candidates.extend(_collision_separation_candidates(obj_b, obj_a, regions))
    if not candidates:
        return None

    chosen = min(
        candidates,
        key=lambda item: _collision_candidate_cost_details(
            move_obj=item["move_obj"],
            candidate_center=item["candidate_center"],
            regions=regions,
            original_center=item["original_center"],
            all_objects=all_objects,
            collision_cfg=collision_cfg,
        )["total_cost"],
    )
    move_obj = chosen["move_obj"]
    keep_obj = chosen["keep_obj"]
    suggested_center_raw = chosen["candidate_center"]
    cost_details = _collision_candidate_cost_details(
        move_obj=move_obj,
        candidate_center=suggested_center_raw,
        regions=regions,
        original_center=chosen["original_center"],
        all_objects=all_objects,
        collision_cfg=collision_cfg,
    )
    outside_penalty = _outside_floor_plan_penalty(move_obj, suggested_center_raw, regions)
    overlap_after = _candidate_overlap_volume(keep_obj, move_obj, suggested_center_raw)
    total_overlap_after, overlap_pairs = _candidate_total_overlap_summary(move_obj, suggested_center_raw, all_objects)
    suggested_center = _round_list(suggested_center_raw)
    vector = _separating_vector(move_obj, keep_obj)
    action = {
        "action": "separate_collision_pair",
        "advisory": True,
        "object_ids": [obj_a.get("object_id"), obj_b.get("object_id")],
        "current_bboxes": [_object_bbox_summary(obj_a), _object_bbox_summary(obj_b)],
        "move_object": move_obj.get("object_id"),
        "anchor_object": keep_obj.get("object_id"),
        "move_object_id": move_obj.get("object_id"),
        "keep_object_id": keep_obj.get("object_id"),
        "separation_axis": vector["overlap_axis"],
        "minimum_delta_m": round(vector["min_separation_distance"], 4),
        "suggested_delta_xy": vector["suggested_delta_xy"],
        "min_separation_distance": round(vector["min_separation_distance"], 4),
        "overlap_axis": vector["overlap_axis"],
        "overlap_depth": round(vector["overlap_depth"], 4),
        "confidence": "medium",
        "candidate_strategy": chosen["strategy"],
        "candidate_overlap_volume_m3": round(overlap_after, 6),
        "candidate_total_overlap_volume_m3": round(total_overlap_after, 6),
        "candidate_overlap_pairs": overlap_pairs,
        "candidate_floor_plan_outside_penalty": round(outside_penalty, 6),
        "cost_mode": cost_details["cost_mode"],
        "outside_cost": round(cost_details["outside_cost"], 6),
        "overlap_cost": round(cost_details["overlap_cost"], 6),
        "movement_cost": round(cost_details["movement_cost"], 6),
        "total_cost": round(cost_details["total_cost"], 6),
        "must_remain_inside_floor_plan": True,
        "collision_pressure": "high" if flag.get("type") == "serious_collision" else "moderate",
        "overlap_ratio": flag.get("overlap_ratio"),
        "intersection_volume_m3": flag.get("intersection_volume_m3"),
        "reason_code": flag.get("type") or "serious_collision",
        "reason": flag.get("message", "serious bbox collision"),
    }
    if total_overlap_after <= 1.0e-9 and outside_penalty <= 1.0e-9:
        action["suggested_center_for_move_object"] = suggested_center
    else:
        action["candidate_center_for_reference"] = suggested_center
        action["candidate_warning"] = (
            "No clean deterministic collision-separation center was found; use this only as a reference and "
            "choose a globally plausible placement with lower total overlap and inside-floor-plan placement."
        )
    return action


def _separating_vector(move_obj: dict, anchor_obj: dict) -> dict:
    center_move = _numeric_triplet(move_obj.get("center")) or [0.0, 0.0, 0.0]
    center_anchor = _numeric_triplet(anchor_obj.get("center")) or [0.0, 0.0, 0.0]
    size_move = _numeric_triplet(move_obj.get("size")) or [0.0, 0.0, 0.0]
    size_anchor = _numeric_triplet(anchor_obj.get("size")) or [0.0, 0.0, 0.0]
    overlaps = []
    for axis in [0, 1]:
        half_sum = (size_move[axis] + size_anchor[axis]) / 2.0
        current_gap = abs(center_move[axis] - center_anchor[axis])
        overlaps.append(max(0.0, half_sum - current_gap))
    axis = 0 if overlaps[0] <= overlaps[1] else 1
    direction = _axis_direction(center_move, center_anchor, axis, str(move_obj.get("object_id")), str(anchor_obj.get("object_id")))
    margin = 0.05
    distance = max(0.05, overlaps[axis] + margin)
    delta = [0.0, 0.0]
    delta[axis] = direction * distance
    return {
        "suggested_delta_xy": _round_list(delta),
        "min_separation_distance": distance,
        "overlap_axis": "x" if axis == 0 else "y",
        "overlap_depth": overlaps[axis],
    }


def _axis_direction(center_move: list[float], center_anchor: list[float], axis: int, move_id: str, anchor_id: str) -> float:
    diff = center_move[axis] - center_anchor[axis]
    if abs(diff) > 1.0e-9:
        return 1.0 if diff > 0 else -1.0
    return 1.0 if move_id >= anchor_id else -1.0


def _height_repair_action(obj: dict, bm_instance: dict, flag: dict, *, reason: str) -> dict:
    center = _numeric_triplet(obj.get("center")) or [0.0, 0.0, 0.0]
    size = _numeric_triplet(obj.get("size")) or [0.0, 0.0, 0.0]
    floor_z = _floor_z(bm_instance)
    wall_height = _room_height_or_none(bm_instance)
    floor_margin = float(flag.get("effective_floor_contact_tolerance_m") or 0.0)
    wall_margin = float(flag.get("effective_above_wall_tolerance_m") or floor_margin)
    min_center_z = floor_z + size[2] / 2.0 + max(0.0, floor_margin)
    max_center_z = (wall_height - size[2] / 2.0 - max(0.0, wall_margin)) if wall_height is not None else None
    if max_center_z is not None and max_center_z < min_center_z:
        return _impossible_height_action(obj, bm_instance, flag)
    high = max_center_z if max_center_z is not None else max(center[2], min_center_z)
    target_z = _clamp(center[2], min_center_z, high)
    suggested = [center[0], center[1], target_z]
    return {
        "action": "adjust_height_within_floor_wall_interval",
        "object_id": obj.get("object_id"),
        "current_bbox": _object_bbox_summary(obj),
        "current_center_z": round(center[2], 4),
        "target_center_z": round(target_z, 4),
        "min_center_z": round(min_center_z, 4),
        "max_center_z": round(max_center_z, 4) if max_center_z is not None else None,
        "floor_z": floor_z,
        "wall_height": wall_height,
        "suggested_center": _round_list(suggested),
        "suggested_delta": _round_list([suggested[i] - center[i] for i in range(3)]),
        "reason": reason,
        "confidence": flag.get("confidence", "medium"),
        "source_kind": flag.get("source_kind"),
        "advisory": True,
    }


def _impossible_height_action(obj: dict, bm_instance: dict, flag: dict) -> dict:
    center = _numeric_triplet(obj.get("center")) or [0.0, 0.0, 0.0]
    size = _numeric_triplet(obj.get("size")) or [0.0, 0.0, 0.0]
    floor_z = _floor_z(bm_instance)
    wall_height = _room_height_or_none(bm_instance)
    return {
        "action": "impossible_height_constraint",
        "code": flag.get("code") or "impossible_height_constraint",
        "object_id": obj.get("object_id"),
        "current_bbox": _object_bbox_summary(obj),
        "object_height": size[2],
        "floor_z": floor_z,
        "wall_height": wall_height,
        "current_center_z": round(center[2], 4),
        "source_kind": flag.get("source_kind"),
        "confidence": flag.get("confidence", "low"),
        "blocking": False,
        "advisory": True,
        "message": flag.get(
            "message",
            "Object height exceeds the available floor-wall interval. Do not attempt naive lowering below floor.",
        ),
    }


def _dense_collision_cluster_actions(physical_flags: list[dict], objects: dict[str, dict], cfg: dict) -> list[dict]:
    pairs = _serious_collision_pairs(physical_flags)
    if not pairs:
        return []
    graph: dict[str, set[str]] = defaultdict(set)
    weights: dict[tuple[str, str], float] = {}
    for a, b, flag in pairs:
        graph[a].add(b)
        graph[b].add(a)
        weights[tuple(sorted([a, b]))] = float(flag.get("overlap_ratio") or 0.0)
    components = _connected_components(graph)
    min_objects = int(cfg.get("dense_cluster_min_objects", 4))
    min_edges = int(cfg.get("dense_cluster_min_edges", 6))
    max_clusters = int(cfg.get("dense_cluster_max_clusters", 5))
    max_pair_cues = int(cfg.get("max_pair_cues_per_cluster", 8))
    actions = []
    for component in components:
        edge_keys = [key for key in weights if key[0] in component and key[1] in component]
        if len(component) < min_objects and len(edge_keys) < min_edges:
            continue
        anchors = _cluster_anchor_objects(component, objects)
        movable = [object_id for object_id in sorted(component) if object_id not in set(anchors)]
        center = _cluster_centroid(component, objects)
        spread = {
            object_id: _round_list(_radial_delta(objects.get(object_id), center, object_id))
            for object_id in movable[: max(0, max_pair_cues)]
        }
        top_edges = sorted(edge_keys, key=lambda key: (-weights[key], key))[:max_pair_cues]
        actions.append(
            {
                "action": "spread_dense_collision_cluster",
                "cluster_id": f"collision_cluster_{len(actions)}",
                "objects": sorted(component),
                "anchor_objects": anchors,
                "movable_objects": movable,
                "suggested_strategy": "radial_spread_from_cluster_centroid",
                "suggested_delta_xy_by_object": spread,
                "top_pair_count": len(top_edges),
                "top_pairs": [list(key) for key in top_edges],
                "omitted_pair_count": max(0, len(edge_keys) - len(top_edges)),
                "confidence": "medium",
                "advisory": True,
                "message": (
                    "This dense cluster has many overlapping objects. Do not move the whole cluster together; "
                    "spread non-anchor objects apart while preserving object ids/sizes."
                ),
            }
        )
        if len(actions) >= max_clusters:
            break
    return actions


def _aggregate_collision_actions(pair_actions: list[dict], cfg: dict) -> list[dict]:
    min_count = int(cfg.get("aggregate_min_collision_count", 2))
    max_magnitude = float(cfg.get("aggregate_max_magnitude_m", 1.5))
    sums: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    counts: dict[str, int] = defaultdict(int)
    partners: dict[str, list[str]] = defaultdict(list)
    for action in pair_actions:
        object_id = action.get("move_object_id")
        delta = action.get("suggested_delta_xy")
        partner = action.get("keep_object_id")
        if not isinstance(object_id, str) or not isinstance(delta, list) or len(delta) < 2:
            continue
        weight = max(float(action.get("overlap_depth") or 0.0), 0.05)
        sums[object_id][0] += float(delta[0]) * weight
        sums[object_id][1] += float(delta[1]) * weight
        counts[object_id] += 1
        if isinstance(partner, str):
            partners[object_id].append(partner)
    actions = []
    for object_id, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        if count < min_count:
            continue
        delta = [sums[object_id][0] / count, sums[object_id][1] / count]
        delta = _cap_xy_delta(delta, max_magnitude)
        actions.append(
            {
                "action": "move_object_to_reduce_collisions",
                "object_id": object_id,
                "suggested_delta_xy": _round_list(delta),
                "collision_count": count,
                "top_partners": sorted(set(partners[object_id]))[:5],
                "contributing_pair_count": count,
                "omitted_pair_count": max(0, count - 5),
                "advisory": True,
                "confidence": "medium",
            }
        )
    return actions


def _serious_collision_pairs(physical_flags: list[dict]) -> list[tuple[str, str, dict]]:
    pairs = []
    for flag in physical_flags:
        if not isinstance(flag, dict) or flag.get("type") != "serious_collision":
            continue
        object_ids = [item for item in flag.get("objects", []) if isinstance(item, str)]
        if len(object_ids) >= 2:
            pairs.append((object_ids[0], object_ids[1], flag))
    return pairs


def _connected_components(graph: dict[str, set[str]]) -> list[set[str]]:
    seen: set[str] = set()
    components = []
    for start in sorted(graph):
        if start in seen:
            continue
        stack = [start]
        component = set()
        while stack:
            item = stack.pop()
            if item in component:
                continue
            component.add(item)
            stack.extend(sorted(graph.get(item, set()) - component))
        seen |= component
        components.append(component)
    components.sort(key=lambda item: (-len(item), sorted(item)))
    return components


def _cluster_anchor_objects(component: set[str], objects: dict[str, dict]) -> list[str]:
    ranked = sorted(component, key=lambda object_id: (-_footprint_area(objects.get(object_id, {})), object_id))
    return ranked[:1]


def _cluster_centroid(component: set[str], objects: dict[str, dict]) -> list[float]:
    centers = [_numeric_triplet(objects.get(object_id, {}).get("center")) for object_id in component]
    centers = [center for center in centers if center is not None]
    if not centers:
        return [0.0, 0.0, 0.0]
    return [sum(center[i] for center in centers) / len(centers) for i in range(3)]


def _radial_delta(obj: dict | None, centroid: list[float], object_id: str) -> list[float]:
    center = _numeric_triplet((obj or {}).get("center"))
    if center is None:
        return [0.0, 0.0]
    dx = center[0] - centroid[0]
    dy = center[1] - centroid[1]
    norm = (dx * dx + dy * dy) ** 0.5
    if norm <= 1.0e-9:
        direction = 1.0 if object_id >= "m" else -1.0
        return [0.35 * direction, 0.0]
    scale = 0.35 / norm
    return [dx * scale, dy * scale]


def _cap_xy_delta(delta: list[float], max_magnitude: float) -> list[float]:
    magnitude = (delta[0] * delta[0] + delta[1] * delta[1]) ** 0.5
    if magnitude <= max_magnitude or magnitude <= 1.0e-9:
        return delta
    scale = max_magnitude / magnitude
    return [delta[0] * scale, delta[1] * scale]


def _footprint_area(obj: dict) -> float:
    size = _numeric_triplet(obj.get("size"))
    return max(0.0, size[0] * size[1]) if size is not None else 0.0


def _collision_separation_candidates(move_obj: dict, keep_obj: dict, regions: list[dict]) -> list[dict]:
    center_move = _numeric_triplet(move_obj.get("center"))
    center_keep = _numeric_triplet(keep_obj.get("center"))
    size_move = _numeric_triplet(move_obj.get("size"))
    size_keep = _numeric_triplet(keep_obj.get("size"))
    if center_move is None or center_keep is None or size_move is None or size_keep is None:
        return []

    margin = 0.05
    candidates = []
    for axis in [0, 1]:
        current_gap = abs(center_move[axis] - center_keep[axis])
        required_gap = (size_move[axis] + size_keep[axis]) / 2.0 + margin
        needed = max(0.05, required_gap - current_gap)
        for direction in [-1.0, 1.0]:
            candidate = list(center_move)
            candidate[axis] += direction * needed
            candidates.append(
                {
                    "move_obj": move_obj,
                    "keep_obj": keep_obj,
                    "axis": axis,
                    "minimum_delta": needed,
                    "candidate_center": candidate,
                    "original_center": center_move,
                    "strategy": "separate",
                }
            )

    inside = _suggest_inside_floor_plan(move_obj, regions).get("suggested_center")
    if isinstance(inside, list) and len(inside) >= 3:
        candidates.append(
            {
                "move_obj": move_obj,
                "keep_obj": keep_obj,
                "axis": 0,
                "minimum_delta": _distance(center_move, inside),
                "candidate_center": [float(inside[0]), float(inside[1]), float(inside[2])],
                "original_center": center_move,
                "strategy": "inside_floor_plan",
            }
        )
    return candidates


def _object_bbox_summary(obj: dict) -> dict:
    return {
        key: obj[key]
        for key in ["object_id", "category", "center", "size", "yaw", "support_parent", "region_id"]
        if key in obj
    }


def _collision_candidate_cost_details(
    *,
    move_obj: dict,
    candidate_center: list[float],
    regions: list[dict],
    original_center: list[float],
    all_objects: dict[str, dict],
    collision_cfg: dict,
) -> dict:
    outside = _outside_floor_plan_penalty(move_obj, candidate_center, regions)
    overlap, overlap_pairs = _candidate_total_overlap_summary(move_obj, candidate_center, all_objects)
    movement = _distance(original_center, candidate_center)
    room_diagonal = _scene_diagonal_m(regions, all_objects)
    move_volume = _object_volume(move_obj)
    max_pair_ratio = max(
        (float(pair.get("overlap_ratio_of_smaller") or 0.0) for pair in overlap_pairs),
        default=0.0,
    )
    overlap_volume_ratio = overlap / max(1.0e-9, move_volume)
    outside_cost = _clamp(outside / room_diagonal, 0.0, 1.0)
    overlap_cost = _clamp(max(max_pair_ratio, overlap_volume_ratio), 0.0, 1.0)
    movement_cost = _clamp(movement / room_diagonal, 0.0, 1.0)
    weights = collision_cfg.get("weights") if isinstance(collision_cfg.get("weights"), dict) else {}
    outside_weight = _positive_or_default(weights.get("outside"), 1.0)
    overlap_weight = _positive_or_default(weights.get("overlap"), 1.0)
    movement_weight = _positive_or_default(weights.get("movement"), 0.25)
    total = outside_weight * outside_cost + overlap_weight * overlap_cost + movement_weight * movement_cost
    return {
        "cost_mode": str(collision_cfg.get("cost_mode") or "normalized_dimensionless"),
        "outside_cost": outside_cost,
        "overlap_cost": overlap_cost,
        "movement_cost": movement_cost,
        "total_cost": total,
        "room_diagonal_m": room_diagonal,
    }


def _candidate_total_overlap_summary(move_obj: dict, candidate_center: list[float], all_objects: dict[str, dict]) -> tuple[float, list[dict]]:
    move_id = move_obj.get("object_id")
    move_volume = _object_volume(move_obj)
    pairs = []
    total = 0.0
    for object_id, other in sorted(all_objects.items()):
        if object_id == move_id:
            continue
        overlap = _candidate_overlap_volume(other, move_obj, candidate_center)
        if overlap <= 1.0e-9:
            continue
        total += overlap
        denominator = min(_object_volume(other), move_volume)
        pairs.append(
            {
                "object_id": object_id,
                "overlap_volume_m3": round(overlap, 6),
                "overlap_ratio_of_smaller": round(overlap / denominator, 6) if denominator > 0 else None,
            }
        )
    pairs.sort(key=lambda item: (-(item.get("overlap_ratio_of_smaller") or 0), item["object_id"]))
    return total, pairs[:8]


def _scene_diagonal_m(regions: list[dict], all_objects: dict[str, dict]) -> float:
    xs: list[float] = []
    ys: list[float] = []
    for region in regions:
        bounds = _region_bounds(region)
        if bounds is None:
            continue
        min_x, max_x, min_y, max_y = bounds
        xs.extend([min_x, max_x])
        ys.extend([min_y, max_y])
    if not xs or not ys:
        for obj in all_objects.values():
            center = _numeric_triplet(obj.get("center"))
            size = _numeric_triplet(obj.get("size"))
            if center is None or size is None:
                continue
            xs.extend([center[0] - size[0] / 2.0, center[0] + size[0] / 2.0])
            ys.extend([center[1] - size[1] / 2.0, center[1] + size[1] / 2.0])
    if not xs or not ys:
        return 1.0
    diagonal = ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5
    return max(diagonal, 1.0)


def _object_volume(obj: dict) -> float:
    size = _numeric_triplet(obj.get("size"))
    if size is None:
        return 0.0
    volume = 1.0
    for value in size:
        volume *= max(0.0, value)
    return volume


def _candidate_overlap_volume(obj_a: dict, obj_b: dict, candidate_center_b: list[float]) -> float:
    center_a = _numeric_triplet(obj_a.get("center"))
    size_a = _numeric_triplet(obj_a.get("size"))
    size_b = _numeric_triplet(obj_b.get("size"))
    if center_a is None or size_a is None or size_b is None:
        return 0.0
    ranges_a = _axis_ranges(center_a, size_a)
    ranges_b = _axis_ranges(candidate_center_b, size_b)
    overlap = 1.0
    for axis in range(3):
        low = max(ranges_a[axis][0], ranges_b[axis][0])
        high = min(ranges_a[axis][1], ranges_b[axis][1])
        if high <= low:
            return 0.0
        overlap *= high - low
    return float(overlap)


def _axis_ranges(center: list[float], size: list[float]) -> list[tuple[float, float]]:
    return [(center[i] - size[i] / 2.0, center[i] + size[i] / 2.0) for i in range(3)]


def _outside_floor_plan_penalty(obj: dict, candidate_center: list[float], regions: list[dict]) -> float:
    size = _numeric_triplet(obj.get("size"))
    if size is None or not regions:
        return 0.0
    region = _matching_region(obj.get("region_id"), regions) or min(
        regions,
        key=lambda item: _distance_to_bounds(candidate_center[0], candidate_center[1], _region_bounds(item) or (0.0, 0.0, 0.0, 0.0)),
    )
    bounds = _region_bounds(region)
    if bounds is None:
        return 0.0
    min_x, max_x, min_y, max_y = bounds
    half_w = size[0] / 2.0
    half_d = size[1] / 2.0
    x_low = min_x + half_w
    x_high = max_x - half_w
    y_low = min_y + half_d
    y_high = max_y - half_d
    penalty = _range_penalty(candidate_center[0], x_low, x_high)
    penalty += _range_penalty(candidate_center[1], y_low, y_high)
    return penalty


def _range_penalty(value: float, low: float, high: float) -> float:
    if low > high:
        midpoint = (low + high) / 2.0
        return abs(value - midpoint) + (low - high)
    if value < low:
        return low - value
    if value > high:
        return value - high
    return 0.0


def _distance(a: list[float], b: list[float]) -> float:
    return sum((float(a[i]) - float(b[i])) ** 2 for i in range(min(len(a), len(b), 3))) ** 0.5


def _matching_region(region_id: object, regions: list[dict]) -> dict | None:
    if not isinstance(region_id, str) or not region_id:
        return None
    for region in regions:
        candidates = {str(region.get(key, "")) for key in ["id", "name", "label"]}
        if region_id in candidates:
            return region
    return None


def _region_bounds(region: dict) -> tuple[float, float, float, float] | None:
    polygon = region.get("floor_polygon")
    if not isinstance(polygon, list) or not polygon:
        return None
    points = []
    for point in polygon:
        if not isinstance(point, list) or len(point) < 2:
            continue
        try:
            points.append((float(point[0]), float(point[1])))
        except (TypeError, ValueError):
            continue
    if not points:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), max(xs), min(ys), max(ys)


def _region_summary(region: dict) -> dict:
    bounds = _region_bounds(region)
    return {
        "id": region.get("id"),
        "label": region.get("label") or region.get("name"),
        "bounds_xy": _round_list(list(bounds)) if bounds is not None else None,
        "source_kind": region.get("source_kind"),
        "source": region.get("source"),
        "synthetic": bool(region.get("synthetic", False)),
    }


def _distance_to_bounds(x: float, y: float, bounds: tuple[float, float, float, float]) -> float:
    min_x, max_x, min_y, max_y = bounds
    dx = 0.0 if min_x <= x <= max_x else min(abs(x - min_x), abs(x - max_x))
    dy = 0.0 if min_y <= y <= max_y else min(abs(y - min_y), abs(y - max_y))
    return dx * dx + dy * dy


def _numeric_triplet(value: object) -> list[float] | None:
    if not isinstance(value, list) or len(value) < 3:
        return None
    try:
        return [float(value[0]), float(value[1]), float(value[2])]
    except (TypeError, ValueError):
        return None


def _room_height(bm_instance: dict) -> float:
    value = _room_height_or_none(bm_instance)
    return value if value is not None else 2.8


def _room_height_or_none(bm_instance: dict) -> float | None:
    room = bm_instance.get("room") if isinstance(bm_instance.get("room"), dict) else {}
    for key in ["height", "wall_height"]:
        try:
            return float(room.get(key))
        except (TypeError, ValueError):
            pass
    return None


def _floor_z(bm_instance: dict) -> float:
    room = bm_instance.get("room") if isinstance(bm_instance.get("room"), dict) else {}
    try:
        return float(room.get("floor_z", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _is_fallback_source(source_kind: object) -> bool:
    text = str(source_kind or "").lower()
    return "fallback" in text or "object_position_extent" in text or text in {"", "unknown"}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _positive_or_default(value: object, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if number >= 0 else default


def _round_list(values: list[float] | None) -> list[float] | None:
    if values is None:
        return None
    return [round(float(value), 4) for value in values]
