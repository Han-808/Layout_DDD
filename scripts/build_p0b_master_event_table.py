#!/usr/bin/env python3
"""Build the audited P0b routed-event master table.

One output row represents one physical `(case_id, metric, event_id)` event. The
four visual configurations are pivoted into columns so all later analyses share
the same event universe. This table intentionally does not invent direct-valid
events that are absent from the experiment artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARMS = (
    "global_raw",
    "visibility_raw",
    "visibility_highlight",
    "visibility_highlight_global",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--camera-events",
        type=Path,
        default=(
            PROJECT_ROOT
            / "Support"
            / "artifacts"
            / "result"
            / "benchmark_metric_analysis"
            / "source_distortion5"
            / "results"
            / "camera_per_event.tsv"
        ),
    )
    parser.add_argument(
        "--gt-audit",
        type=Path,
        default=PROJECT_ROOT / "Support" / "datasets" / "cal_dataset0" / "validation" / "GT_COORDINATE_AUDIT.tsv",
    )
    parser.add_argument(
        "--event-root",
        type=Path,
        default=(
            PROJECT_ROOT
            / "Support"
            / "artifacts"
            / "result"
            / "benchmark_metric_analysis"
            / "source_distortion5"
            / "ablation"
        ),
    )
    parser.add_argument(
        "--picture-mapping",
        type=Path,
        default=(
            PROJECT_ROOT
            / "Support"
            / "artifacts"
            / "result"
            / "benchmark_metric_analysis"
            / "picture_mapping"
            / "judge_picture_mapping.csv"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=(
            PROJECT_ROOT
            / "Support"
            / "artifacts"
            / "result"
            / "benchmark_metric_analysis"
            / "master_event_table.tsv"
        ),
    )
    parser.add_argument(
        "--validation-out",
        type=Path,
        default=(
            PROJECT_ROOT
            / "Support"
            / "artifacts"
            / "result"
            / "benchmark_metric_analysis"
            / "master_event_table_validation.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    camera_rows = _read_tsv(args.camera_events)
    audit_rows = {
        _key(row): row
        for row in _read_tsv(args.gt_audit)
    }
    packets = _load_global_packets(args.event_root)
    picture_counts = _load_picture_counts(args.picture_mapping)

    grouped: dict[tuple[str, str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in camera_rows:
        key = _key(row)
        arm = row["arm"]
        if arm in grouped[key]:
            raise RuntimeError(f"duplicate event/arm row: {key}/{arm}")
        grouped[key][arm] = row

    camera_keys = set(grouped)
    if camera_keys != set(audit_rows):
        raise RuntimeError(_set_difference_message("camera events", camera_keys, "GT audit", set(audit_rows)))
    if camera_keys != set(packets):
        raise RuntimeError(_set_difference_message("camera events", camera_keys, "global packets", set(packets)))

    rows: list[dict[str, Any]] = []
    for key in sorted(camera_keys):
        case_id, metric, event_id = key
        arm_rows = grouped[key]
        missing_arms = set(ARMS) - set(arm_rows)
        extra_arms = set(arm_rows) - set(ARMS)
        if missing_arms or extra_arms:
            raise RuntimeError(
                f"{key}: arm mismatch; missing={sorted(missing_arms)}, extra={sorted(extra_arms)}"
            )

        audit = audit_rows[key]
        packet_path, packet = packets[key]
        request = (packet.get("judgement") or {}).get("request") or {}
        detector = request.get("detector_evidence") or {}
        object_ids = packet.get("object_ids") or (request.get("event") or {}).get("object_ids") or []

        rendered_label = audit["rendered_geometry_classification"]
        if metric in {"collision", "oob"}:
            gt_label = rendered_label
            gt_status = "usable_rendered_mesh_gt"
        else:
            gt_label = ""
            gt_status = "blocked_pending_independent_transformed_mesh_gt"

        row: dict[str, Any] = {
            "physical_event_key": "::".join(key),
            "case_id": case_id,
            "base_case_id": arm_rows[ARMS[0]]["base_case_id"],
            "distortion_family": arm_rows[ARMS[0]]["family"],
            "metric": metric,
            "event_id": event_id,
            "object_ids": json.dumps(object_ids, separators=(",", ":")),
            "gt_label": gt_label,
            "gt_validity_status": gt_status,
            "stored_gt_label": audit["stored_gt_label"],
            "canonical_proxy_label": audit["canonical_coordinate_classification"],
            "provisional_rendered_mesh_label": rendered_label,
            "gt_audit_outcome": audit["audit_outcome"],
            "detector_result": _detector_summary(metric, detector),
            "detector_backend": _detector_backend(metric, detector),
            "detector_evidence_level": _detector_evidence_level(metric, detector),
            "candidate_selection_policy": detector.get("candidate_selection_policy", ""),
            "routing_status": "routed_to_vlm",
            "universe_scope": "routed_candidates_only",
            "detector_event_packet": str(packet_path.relative_to(PROJECT_ROOT)),
        }

        resolved_predictions: list[str] = []
        for arm in ARMS:
            source = arm_rows[arm]
            resolved = _bool(source["resolved"])
            prediction = source["predicted_label"] if resolved else ""
            if prediction:
                resolved_predictions.append(prediction)
            error = source["error"]
            row.update(
                {
                    f"{arm}_prediction": prediction,
                    f"{arm}_resolved": resolved,
                    f"{arm}_matches_gt": (
                        prediction == gt_label if resolved and gt_label else ""
                    ),
                    f"{arm}_evidence_failure": bool(error),
                    f"{arm}_render_failure": error.startswith("BlenderRenderError:"),
                    f"{arm}_failure_type": _failure_type(error),
                    f"{arm}_error": error,
                    f"{arm}_confidence": _number(source["confidence"]),
                    f"{arm}_image_count": _integer(source["image_count"]),
                    f"{arm}_mapped_judge_images": picture_counts.get((*key, arm), 0),
                    f"{arm}_camera_seconds": _number(source["camera_evidence_seconds"]),
                    f"{arm}_judge_seconds": _number(source["judge_seconds"]),
                    f"{arm}_elapsed_seconds": _number(source["elapsed_seconds"]),
                    f"{arm}_estimated_uncached_seconds": _number(
                        source["estimated_uncached_seconds"]
                    ),
                }
            )

        row["any_unresolved"] = any(not _bool(arm_rows[arm]["resolved"]) for arm in ARMS)
        row["resolved_verdict_flip"] = len(set(resolved_predictions)) > 1
        row["resolved_prediction_pattern"] = "|".join(
            f"{arm}={row[f'{arm}_prediction'] or 'unresolved'}" for arm in ARMS
        )
        rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    _write_tsv(args.out, rows)
    validation = _validate(rows, camera_rows, picture_counts)
    args.validation_out.write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")

    print(f"physical_events={len(rows)}")
    print(f"event_config_packets={len(camera_rows)}")
    print(f"master_table={args.out}")
    print(f"validation={args.validation_out}")


def _load_global_packets(root: Path) -> dict[tuple[str, str, str], tuple[Path, dict[str, Any]]]:
    packets: dict[tuple[str, str, str], tuple[Path, dict[str, Any]]] = {}
    for path in sorted(root.glob("*/global_raw/events/*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        case_id = path.parents[2].name
        key = (case_id, str(value["metric"]), str(value["event_id"]))
        if key in packets:
            raise RuntimeError(f"duplicate global packet: {key}")
        packets[key] = (path, value)
    return packets


def _load_picture_counts(path: Path) -> dict[tuple[str, str, str, str], int]:
    counts: dict[tuple[str, str, str, str], int] = defaultdict(int)
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            # The mapping keeps one pathless placeholder for every unresolved
            # packet. It is an event record, not a judge-image reference.
            if not row.get("remote_image_path"):
                continue
            key = (row["case_id"], row["metric"], row["event_id"], row["arm"])
            counts[key] += 1
    return dict(counts)


def _detector_summary(metric: str, detector: dict[str, Any]) -> str:
    if metric == "collision":
        obb = detector.get("obb") or {}
        mesh = detector.get("mesh") or {}
        summary = {
            "obb_intersects": obb.get("intersects"),
            "obb_overlap_depth_proxy_m": obb.get("minimum_overlap_depth_proxy_m"),
            "mesh_state": mesh.get("mesh_state"),
            "mesh_surface_intersection": mesh.get("surface_intersection"),
            "mesh_minimum_surface_distance_m": mesh.get("minimum_surface_distance_m"),
        }
    elif metric == "oob":
        flags = detector.get("plane_flags") or {}
        summary = {
            "violated_planes": sorted(
                key.removesuffix("_oob") for key, value in flags.items() if value
            ),
            "obb_intervals": detector.get("obb_intervals"),
        }
    elif metric == "support":
        summary = {
            "gap_band": detector.get("gap_band"),
            "minimum_positive_clearance_m": detector.get("minimum_positive_clearance_m"),
            "near_support_tolerance_m": detector.get("near_support_tolerance_m"),
            "base_contact_hit_count": detector.get("base_contact_hit_count"),
            "base_contact_sample_count": detector.get("base_contact_sample_count"),
            "measured_support_modes": detector.get("measured_support_modes"),
            "geometry_evidence_degraded": detector.get("geometry_evidence_degraded"),
            "routing_reasons": detector.get("routing_reasons"),
        }
    else:
        summary = detector
    return json.dumps(summary, separators=(",", ":"), sort_keys=True)


def _detector_backend(metric: str, detector: dict[str, Any]) -> str:
    if metric == "collision":
        mesh = detector.get("mesh") or {}
        obb = detector.get("obb") or {}
        return "+".join(filter(None, [str(obb.get("backend") or ""), str(mesh.get("backend") or "")]))
    return str(detector.get("detector") or "")


def _detector_evidence_level(metric: str, detector: dict[str, Any]) -> str:
    if metric == "collision":
        return "mesh" if detector.get("mesh") else "obb"
    return str(detector.get("evidence_level") or "")


def _failure_type(error: str) -> str:
    if not error:
        return ""
    if error.startswith("BlenderRenderError:"):
        return "blank_or_near_uniform_render"
    if "produced no evidence for style 'highlight'" in error:
        return "missing_highlight_evidence"
    return "other_evidence_failure"


def _validate(
    rows: list[dict[str, Any]],
    camera_rows: list[dict[str, str]],
    picture_counts: dict[tuple[str, str, str, str], int],
) -> dict[str, Any]:
    by_metric: dict[str, int] = defaultdict(int)
    gt_status: dict[str, int] = defaultdict(int)
    for row in rows:
        by_metric[str(row["metric"])] += 1
        gt_status[str(row["gt_validity_status"])] += 1

    mapped_references = sum(picture_counts.values())
    expected_references = sum(_integer(row["image_count"]) or 0 for row in camera_rows)
    if mapped_references != expected_references:
        raise RuntimeError(
            f"judge-image mapping mismatch: mapped={mapped_references}, expected={expected_references}"
        )

    return {
        "status": "valid_with_known_scope_limitations",
        "physical_event_rows": len(rows),
        "event_config_packets": len(camera_rows),
        "visual_configs_per_event": len(ARMS),
        "events_by_metric": dict(sorted(by_metric.items())),
        "gt_validity_status": dict(sorted(gt_status.items())),
        "routed_events": sum(row["routing_status"] == "routed_to_vlm" for row in rows),
        "direct_valid_events_present": 0,
        "universe_scope": "routed_candidates_only",
        "judge_image_references": mapped_references,
        "events_with_any_unresolved_config": sum(bool(row["any_unresolved"]) for row in rows),
        "events_with_resolved_verdict_flip": sum(
            bool(row["resolved_verdict_flip"]) for row in rows
        ),
        "support_gt_note": (
            "Support gt_label is intentionally empty. provisional_rendered_mesh_label is retained "
            "for diagnosis but is not frozen until an independent final transformed-mesh GT audit."
        ),
    }


def _key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row["case_id"], row["metric"], row["event_id"])


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("cannot write an empty master event table")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _number(value: str) -> float | None:
    return float(value) if value not in {"", None} else None


def _integer(value: str) -> int | None:
    return int(value) if value not in {"", None} else None


def _set_difference_message(
    left_name: str,
    left: set[tuple[str, str, str]],
    right_name: str,
    right: set[tuple[str, str, str]],
) -> str:
    return (
        f"event universe mismatch between {left_name} and {right_name}; "
        f"only_{left_name}={sorted(left - right)}, only_{right_name}={sorted(right - left)}"
    )


if __name__ == "__main__":
    main()
