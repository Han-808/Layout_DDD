"""Workflow-only coverage checks and diagnostic grouping; never score or judge.

Repeated signatures are a triage signal, not proof of a shared root cause.
The publication check certifies completeness only, not benchmark comparability.
"""
from __future__ import annotations

import math

from .safety import IntegrityError, redact

POLICY = "complete_original_reports_only_v1"
STATUSES = ("complete", "skipped_hard_failure", "infra_failure", "unresolved")


def derive(value: dict, metrics: tuple[str, ...]) -> dict:
    scenes = value["scenes"]
    rows = [r for s in scenes for r in s["rooms"]]
    planned = value["planned_room_count"]
    histogram = {}
    for room in rows:
        if room["status"] == "complete":
            continue
        details = list(room.get("failure_details") or [])
        for metric in room["metrics"].values():
            if metric["status"] != "complete":
                details.extend(metric.get("failure_details") or [])
        if not details:
            failure = room.get("failure") or {}
            details = [{"stage": failure.get("stage"), "metric": None,
                        "reason_code": "category." + str(failure.get("category", room["status"])),
                        "error_type": failure.get("error_type")}]
        for detail in details:
            # Never group by raw messages, prompts, ID inventories or scores.
            fields = ("stage", "metric", "reason_code", "error_type")
            key = tuple(redact(str(detail.get(k) or "unknown"))[:256] for k in fields)
            group = histogram.setdefault(key, {**dict(zip(fields, key)), "rooms": set()})
            group["rooms"].add((room["model"], room["scene_id"], room["room_id"]))
    groups = []
    for _, group in sorted(histogram.items()):
        affected = sorted(group["rooms"])
        groups.append({**{k: v for k, v in group.items() if k != "rooms"},
                       "affected_room_count": len(affected), "planned_room_count": planned,
                       "examples": [dict(zip(("model", "scene_id", "room_id"), r)) for r in affected[:8]],
                       "examples_truncated": len(affected) > 8,
                       "same_root_cause_confirmed": False})
    blockers = []
    if not planned: blockers.append("empty_planned_inventory")
    if not value["all_work_accounted"]: blockers.append("execution_not_fully_accounted")
    if value["score_coverage"]["complete_rooms"] != planned:
        blockers.append("room_score_coverage_incomplete")
    if value["score_coverage"]["completed_metrics"] != planned * len(metrics):
        blockers.append("metric_score_coverage_incomplete")
    if not scenes or any(not s["rooms"] or s["official_aggregation_status"] != "complete" for s in scenes):
        blockers.append("original_scene_aggregates_incomplete")
    if value.get("execution_failure") is not None: blockers.append("execution_aborted")
    alerts = []
    if value["room_counts"]["skipped_hard_failure"]:
        alerts.append({"code": "hard_failure_skips_present", "severity": "warning",
                       "affected_room_count": value["room_counts"]["skipped_hard_failure"],
                       "planned_room_count": planned})
    if value["aggregation_skip_count"]:
        alerts.append({"code": "scene_aggregation_skips_present", "severity": "warning",
                       "affected_scene_count": value["aggregation_skip_count"]})
    for group in groups:
        if group["affected_room_count"] > 1:
            alerts.append({"code": "repeated_failure_signature", "severity": "warning", **group})
    return {"failure_reason_summary": groups, "workflow_alerts": alerts,
            "metric_coverage": {name: {"complete_rooms": sum(r["metrics"][name]["status"] == "complete" for r in rows),
                                       "planned_rooms": planned} for name in metrics},
            "official_score_publication": {"policy": POLICY, "eligible": not blockers,
                "scope": "execution_completeness_only_not_quality_or_comparability", "blockers": blockers}}


def validate(value: dict, metrics: tuple[str, ...]) -> None:
    """Reconcile the envelope before a parent or publication check trusts it."""
    planned = value.get("planned_room_count")
    if type(planned) is not int or planned < 0:
        raise IntegrityError("invalid planned execution inventory")
    seen, scenes_seen, rows = set(), set(), []
    for scene in value["scenes"]:
        scene_key = (scene["model"], scene["scene_id"])
        if any(not isinstance(k, str) or not k or k in {".", ".."} or "/" in k or "\\" in k for k in scene_key):
            raise IntegrityError("invalid scene scope in execution result")
        if scene_key in scenes_seen or scene["official_aggregation_status"] not in {"complete", "skipped", "not_available"}:
            raise IntegrityError("invalid scene execution inventory")
        scenes_seen.add(scene_key)
        for room in scene["rooms"]:
            key = (room["model"], room["scene_id"], room["room_id"])
            if (not isinstance(key[2], str) or not key[2] or key[2] in {".", ".."} or "/" in key[2] or "\\" in key[2]
                    or key in seen or key[:2] != scene_key or room["status"] not in STATUSES):
                raise IntegrityError("invalid or duplicate room execution inventory")
            seen.add(key)
            if set(room["metrics"]) != set(metrics) or room.get("score") is not None:
                raise IntegrityError("execution metrics cannot omit required metrics or invent a room score")
            complete = 0
            for name, metric in room["metrics"].items():
                score = metric.get("score")
                if metric["status"] == "complete":
                    if (metric.get("metric") != name or isinstance(score, bool) or not isinstance(score, (int, float))
                            or not math.isfinite(score) or not 0 <= score <= 1):
                        raise IntegrityError("invalid complete metric in execution envelope")
                    complete += 1
                elif (score is not None or metric["status"] not in {
                        "skipped", "hard_failure", "infra_failure", "unresolved", "not_evaluated"}):
                    raise IntegrityError("unscored work cannot acquire a numerical score")
            if (type(room.get("completed_metric_count")) is not int or type(room.get("required_metric_count")) is not int
                    or room["completed_metric_count"] != complete or room["required_metric_count"] != len(metrics)):
                raise IntegrityError("room metric counters do not reconcile")
            if room["status"] == "complete" and (complete != len(metrics) or not room.get("official_room_report")):
                raise IntegrityError("complete execution room requires its original complete report")
            if room["status"] == "skipped_hard_failure" and any(
                    m["status"] not in {"complete", "skipped"} for m in room["metrics"].values()):
                raise IntegrityError("hard skip cannot absorb unresolved infrastructure")
            rows.append(room)
    if len(rows) > planned:
        raise IntegrityError("observed execution inventory exceeds plan")
    counts = {k: sum(r["status"] == k for r in rows) for k in STATUSES}
    coverage = {"complete_rooms": counts["complete"], "planned_rooms": planned,
                "completed_metrics": sum(r["completed_metric_count"] for r in rows),
                "planned_metrics": planned * len(metrics)}
    for name in ("room_counts", "score_coverage"):
        if not isinstance(value.get(name), dict) or any(type(n) is not int or n < 0 for n in value[name].values()):
            raise IntegrityError("execution coverage requires non-negative integer counters")
    missing = planned - len(rows)
    skips = sum(s["official_aggregation_status"] == "skipped" for s in value["scenes"])
    unmapped = sum(not s["rooms"] for s in value["scenes"])
    unaggregated = any(s["rooms"] and all(r["status"] == "complete" for r in s["rooms"])
                      and s["official_aggregation_status"] == "not_available" for s in value["scenes"])
    accounted = (not missing and not counts["infra_failure"] and not counts["unresolved"]
                 and not unmapped and not unaggregated and value.get("execution_failure") is None)
    status = ("completed_with_skips" if counts["skipped_hard_failure"] or skips else "complete") if accounted else "incomplete"
    for name, expected in {"room_counts": counts, "score_coverage": coverage, "observed_room_count": len(rows),
                           "not_started_room_count": missing, "aggregation_skip_count": skips,
                           "unmapped_scene_count": unmapped, "all_work_accounted": accounted,
                           "status": status, "aggregate_score": None, "skipped_items_are_not_scores": True,
                           **derive(value, metrics)}.items():
        if value.get(name) != expected or (type(expected) in {int, bool} and type(value.get(name)) is not type(expected)):
            raise IntegrityError(f"execution result does not reconcile: {name}")
