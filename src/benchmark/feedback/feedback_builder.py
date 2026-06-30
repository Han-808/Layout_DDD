from __future__ import annotations

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


def build_feedback(
    evaluation_report: dict,
    current_layout: dict,
    bm_instance: dict,
    benchmark_config: dict | None = None,
) -> dict:
    collision_cfg = _collision_avoidance_config(benchmark_config)
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
    violations.extend(_debug_flags_to_violations("physical_debug_flag", physical_flags))
    violations.extend(_vlm_issues_to_violations(evaluation_report.get("vlm_judgement", {}).get("issues", [])))
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

    return {
        "task_id": evaluation_report.get("task_id", bm_instance.get("task_id", "unknown_task")),
        "iteration": int(evaluation_report.get("iteration", 0)),
        "repair_targets": repair_targets,
        "locked_objects": locked_objects,
        "violations": violations,
        "repair_actions": _repair_actions(
            repair_targets=repair_targets,
            current_layout=current_layout,
            bm_instance=bm_instance,
            physical_flags=physical_flags,
            vlm_issues=evaluation_report.get("vlm_judgement", {}).get("issues", []),
            soft_collision_pairs=soft_collision_pairs,
            collision_cfg=collision_cfg,
        ),
        "debug_evidence_summary": _debug_evidence_summary(debug_evidence),
        "room_consistency_reason": room_consistency.get("short_reason", ""),
        "instruction": "Fix the listed violations and debug physical flags. Preserve valid objects. Return corrected layout JSON only.",
    }


def _collision_avoidance_config(benchmark_config: dict | None) -> dict:
    repair = benchmark_config.get("repair", {}) if isinstance(benchmark_config, dict) else {}
    override = repair.get("collision_avoidance", {}) if isinstance(repair, dict) else {}
    if not isinstance(override, dict):
        override = {}
    return _merge_collision_config(DEFAULT_COLLISION_AVOIDANCE_CONFIG, override)


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
        flag_type = flag.get("type")
        if flag_type not in {"room_boundary", "below_floor", "above_wall_height", "serious_collision"}:
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
        flag_type = flag.get("type")
        if flag_type not in {"room_boundary", "below_floor", "above_wall_height", "serious_collision"}:
            continue
        violations.append(
            {
                "category": category,
                "type": flag_type,
                "message": flag.get("message", ""),
                "objects": [item for item in flag.get("objects", []) if isinstance(item, str)],
                "severity": flag.get("severity"),
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
) -> list[dict]:
    objects = _layout_object_map(current_layout)
    regions = _floor_plan_regions(bm_instance)
    actions: list[dict] = []
    physical_flag_list = physical_flags if isinstance(physical_flags, list) else []
    for flag in physical_flag_list:
        if not isinstance(flag, dict):
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
                        "action": "move_inside_floor_plan",
                        "object_id": object_id,
                        "current_bbox": _object_bbox_summary(obj),
                        "target_region": suggestion.get("target_region"),
                        "suggested_center": suggestion.get("suggested_center"),
                        "suggested_delta": suggestion.get("suggested_delta"),
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
                suggested = [center[0], center[1], max(center[2], size[2] / 2.0)]
                actions.append(
                    {
                        "action": "raise_above_floor",
                        "object_id": object_id,
                        "current_bbox": _object_bbox_summary(obj),
                        "suggested_center": _round_list(suggested),
                        "suggested_delta": _round_list([suggested[i] - center[i] for i in range(3)]),
                        "reason": flag.get("message", "object bottom is below floor"),
                    }
                )
        elif flag_type == "above_wall_height":
            for object_id in object_ids:
                obj = objects.get(object_id)
                if obj is None:
                    continue
                center = _numeric_triplet(obj.get("center"))
                size = _numeric_triplet(obj.get("size"))
                if center is None or size is None:
                    continue
                wall_height = _room_height(bm_instance)
                suggested = [center[0], center[1], min(center[2], wall_height - size[2] / 2.0)]
                actions.append(
                    {
                        "action": "lower_below_wall_height",
                        "object_id": object_id,
                        "current_bbox": _object_bbox_summary(obj),
                        "wall_height": wall_height,
                        "suggested_center": _round_list(suggested),
                        "suggested_delta": _round_list([suggested[i] - center[i] for i in range(3)]),
                        "reason": flag.get("message", "object top is above wall height"),
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
                    "severity",
                    "objects",
                    "message",
                    "group_id",
                    "projection",
                    "view_id",
                    "overlap_ratio",
                    "threshold",
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
    return [region for region in regions if isinstance(region, dict) and _region_bounds(region) is not None]


def _suggest_inside_floor_plan(obj: dict, regions: list[dict]) -> dict:
    center = _numeric_triplet(obj.get("center"))
    size = _numeric_triplet(obj.get("size"))
    if center is None or size is None or not regions:
        return {"target_region": None, "suggested_center": center, "suggested_delta": [0.0, 0.0, 0.0]}

    region = _matching_region(obj.get("region_id"), regions) or min(
        regions,
        key=lambda item: _distance_to_bounds(center[0], center[1], _region_bounds(item) or (0.0, 0.0, 0.0, 0.0)),
    )
    bounds = _region_bounds(region)
    if bounds is None:
        return {"target_region": None, "suggested_center": _round_list(center), "suggested_delta": [0.0, 0.0, 0.0]}

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
    return {
        "target_region": _region_summary(region),
        "suggested_center": _round_list(suggested),
        "suggested_delta": _round_list([suggested[i] - center[i] for i in range(3)]),
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
    action = {
        "action": "separate_collision_pair",
        "object_ids": [obj_a.get("object_id"), obj_b.get("object_id")],
        "current_bboxes": [_object_bbox_summary(obj_a), _object_bbox_summary(obj_b)],
        "move_object_id": move_obj.get("object_id"),
        "keep_object_id": keep_obj.get("object_id"),
        "separation_axis": "x" if chosen["axis"] == 0 else "y",
        "minimum_delta_m": round(chosen["minimum_delta"], 4),
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
    room = bm_instance.get("room") if isinstance(bm_instance.get("room"), dict) else {}
    for key in ["height", "wall_height"]:
        try:
            return float(room.get(key))
        except (TypeError, ValueError):
            pass
    return 2.8


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
