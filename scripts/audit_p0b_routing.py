#!/usr/bin/env python3
"""Audit P0b detector routing without visual or manual review."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-root",
        type=Path,
        default=(
            PROJECT_ROOT
            / "result"
            / "benchmark_metric_analysis"
            / "source_distortion5"
            / "source_reports"
        ),
    )
    parser.add_argument(
        "--fixture-root",
        type=Path,
        default=PROJECT_ROOT / "Support" / "datasets" / "cal_dataset0" / "controlled_distortions" / "fixtures",
    )
    parser.add_argument(
        "--gt-audit",
        type=Path,
        default=PROJECT_ROOT / "Support" / "datasets" / "cal_dataset0" / "validation" / "GT_COORDINATE_AUDIT.tsv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "Support" / "artifacts" / "result" / "benchmark_metric_analysis" / "routing_audit.tsv",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=PROJECT_ROOT / "Support" / "artifacts" / "result" / "benchmark_metric_analysis" / "routing_audit_summary.json",
    )
    parser.add_argument(
        "--flags-out",
        type=Path,
        default=PROJECT_ROOT / "Support" / "artifacts" / "result" / "benchmark_metric_analysis" / "routing_flags.tsv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audited = {
        (row["case_id"], row["metric"], row["event_id"]): row
        for row in _read_tsv(args.gt_audit)
    }
    rows: list[dict[str, Any]] = []

    for report_path in sorted(args.report_root.glob("*/evaluation_report.json")):
        case_id = report_path.parent.name
        report = _read_json(report_path)
        manifest = _read_json(args.fixture_root / case_id / "distortion_manifest.json")
        metrics = report["reports"]["generic_validity"]["metrics"]
        expected = manifest["expected"]

        collision_invalid = {
            "|".join(sorted(str(item) for item in pair))
            for pair in expected["collision_invalid_pairs"]
        }
        oob_invalid = {str(item) for item in expected["oob_invalid_object_ids"]}

        for record in metrics["collision"]["pairs"]:
            event_id = "|".join(sorted((str(record["object_a"]), str(record["object_b"]))))
            key = (case_id, "collision", event_id)
            audit = audited.get(key)
            if audit is not None:
                gt_label = audit["rendered_geometry_classification"]
                gt_basis = "independent_rendered_mesh_collision_audit"
            else:
                gt_label = "invalid" if event_id in collision_invalid else "valid"
                gt_basis = (
                    "controlled_transform"
                    if event_id in collision_invalid
                    else "certified_obb_separation_under_uniform_containment_contract"
                )
            rows.append(
                _collision_row(case_id, manifest, record, event_id, gt_label, gt_basis)
            )

        for record in metrics["oob"]["objects"]:
            event_id = str(record["object_id"])
            key = (case_id, "oob", event_id)
            audit = audited.get(key)
            gt_label = "invalid" if event_id in oob_invalid else "valid"
            gt_basis = (
                "independent_rendered_mesh_oob_audit"
                if audit is not None
                else "controlled_fixture_and_inside_obb_under_uniform_containment_contract"
            )
            if audit is not None and audit["rendered_geometry_classification"] != gt_label:
                raise RuntimeError(f"OOB GT disagreement for {key}")
            rows.append(_oob_row(case_id, manifest, record, event_id, gt_label, gt_basis))

        for record in metrics["support"]["objects"]:
            event_id = str(record["object_id"])
            audit = audited.get((case_id, "support", event_id))
            provisional_label = (
                audit["rendered_geometry_classification"]
                if audit is not None
                else ("valid" if record["route"] == "direct_valid_contact" else "")
            )
            rows.append(
                _support_row(case_id, manifest, record, event_id, provisional_label)
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    _write_tsv(args.out, rows)
    flags = [row for row in rows if row["incorrect_routing"] or row["efficiency_flag"]]
    _write_tsv(args.flags_out, flags, fieldnames=list(rows[0]))
    summary = _summarize(rows)
    args.summary_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"routing_records={len(rows)}")
    print(f"incorrect_routing={summary['incorrect_routing_records']}")
    print(f"efficiency_flags={summary['efficiency_flag_records']}")
    print(f"audit={args.out}")
    print(f"flags={args.flags_out}")
    print(f"summary={args.summary_out}")


def _collision_row(
    case_id: str,
    manifest: dict[str, Any],
    record: dict[str, Any],
    event_id: str,
    gt_label: str,
    gt_basis: str,
) -> dict[str, Any]:
    route = str(record["route"])
    routed = bool(record["requires_vlm"])
    mesh = record.get("mesh_evidence") or {}
    if gt_label == "invalid" and routed:
        assessment = "correct_route"
        incorrect = False
        efficiency = False
    elif gt_label == "invalid" and not routed:
        assessment = "missed_route"
        incorrect = True
        efficiency = False
    elif gt_label == "valid" and not routed:
        assessment = "correct_bypass"
        incorrect = False
        efficiency = False
    elif _collision_route_is_conservative(mesh):
        assessment = "justified_conservative_route"
        incorrect = False
        efficiency = True
    else:
        assessment = "unnecessary_route"
        incorrect = True
        efficiency = True
    obb = record.get("obb_evidence") or {}
    detector_summary = {
        "obb_intersects": obb.get("intersects"),
        "obb_overlap_depth_proxy_m": obb.get("minimum_overlap_depth_proxy_m"),
        "mesh_state": mesh.get("mesh_state"),
        "mesh_reliable_for_separation": mesh.get("mesh_reliable_for_separation"),
        "mesh_surface_intersection": mesh.get("surface_intersection"),
        "mesh_minimum_surface_distance_m": mesh.get("minimum_surface_distance_m"),
    }
    return _base_row(
        case_id,
        manifest,
        "collision",
        event_id,
        [record["object_a"], record["object_b"]],
        gt_label,
        "usable",
        gt_basis,
        route,
        routed,
        assessment,
        incorrect,
        efficiency,
        detector_summary,
        record,
    )


def _oob_row(
    case_id: str,
    manifest: dict[str, Any],
    record: dict[str, Any],
    event_id: str,
    gt_label: str,
    gt_basis: str,
) -> dict[str, Any]:
    route = str(record["route"])
    routed = bool(record["requires_vlm"])
    if gt_label == "invalid" and routed:
        assessment = "correct_route"
        incorrect = False
        efficiency = False
    elif gt_label == "invalid":
        assessment = "missed_route"
        incorrect = True
        efficiency = False
    elif not routed:
        assessment = "correct_bypass"
        incorrect = False
        efficiency = False
    else:
        assessment = "unnecessary_route"
        incorrect = True
        efficiency = True
    detector_summary = {
        "candidate_oob": record.get("candidate_oob"),
        "violated_planes": sorted(
            key.removesuffix("_oob")
            for key, value in (record.get("plane_flags") or {}).items()
            if value
        ),
        "obb_intervals": record.get("obb_intervals"),
    }
    return _base_row(
        case_id,
        manifest,
        "oob",
        event_id,
        [record["object_id"]],
        gt_label,
        "usable",
        gt_basis,
        route,
        routed,
        assessment,
        incorrect,
        efficiency,
        detector_summary,
        record,
    )


def _support_row(
    case_id: str,
    manifest: dict[str, Any],
    record: dict[str, Any],
    event_id: str,
    provisional_label: str,
) -> dict[str, Any]:
    route = str(record["route"])
    routed = bool(record["requires_vlm"])
    reliable_contact = (
        int(record.get("base_contact_hit_count") or 0) >= 1
        and not bool(record.get("geometry_evidence_degraded"))
        and bool(record.get("certified_grounded_support"))
    )
    suspicious = bool(record.get("geometry_evidence_degraded")) or (
        record.get("minimum_positive_clearance_m") is not None
        and float(record["minimum_positive_clearance_m"]) > 0.0
    ) or not bool(record.get("certified_grounded_support"))
    if route == "direct_valid_contact" and reliable_contact and not routed:
        assessment = "policy_consistent_direct_contact"
        incorrect = False
    elif route == "vlm_adjudicated" and suspicious and routed:
        assessment = "policy_consistent_suspicious_route"
        incorrect = False
    else:
        assessment = "routing_policy_violation"
        incorrect = True
    detector_summary = {
        "gap_band": record.get("gap_band"),
        "minimum_positive_clearance_m": record.get("minimum_positive_clearance_m"),
        "near_support_tolerance_m": record.get("near_support_tolerance_m"),
        "base_contact_hit_count": record.get("base_contact_hit_count"),
        "base_contact_sample_count": record.get("base_contact_sample_count"),
        "geometry_evidence_degraded": record.get("geometry_evidence_degraded"),
        "certified_grounded_support": record.get("certified_grounded_support"),
        "grounding_status": record.get("grounding_status"),
        "grounded_support_path": record.get("grounded_support_path"),
        "ungrounded_contact_cycle_reachable": record.get(
            "ungrounded_contact_cycle_reachable"
        ),
        "routing_reasons": record.get("routing_reasons"),
    }
    return _base_row(
        case_id,
        manifest,
        "support",
        event_id,
        [record["object_id"]],
        "",
        "blocked_pending_independent_transformed_mesh_gt",
        "provisional_rendered_label=" + provisional_label if provisional_label else "",
        route,
        routed,
        assessment,
        incorrect,
        False,
        detector_summary,
        record,
    )


def _base_row(
    case_id: str,
    manifest: dict[str, Any],
    metric: str,
    event_id: str,
    object_ids: list[str],
    gt_label: str,
    gt_status: str,
    gt_basis: str,
    route: str,
    routed: bool,
    assessment: str,
    incorrect: bool,
    efficiency: bool,
    detector_summary: dict[str, Any],
    record: dict[str, Any],
) -> dict[str, Any]:
    return {
        "routing_event_key": f"{case_id}::{metric}::{event_id}",
        "case_id": case_id,
        "base_case_id": manifest["base_case_id"],
        "distortion_family": manifest["family"],
        "metric": metric,
        "event_id": event_id,
        "object_ids": json.dumps(object_ids, separators=(",", ":")),
        "gt_label": gt_label,
        "gt_status": gt_status,
        "gt_basis": gt_basis,
        "route": route,
        "requires_vlm": routed,
        "routing_assessment": assessment,
        "incorrect_routing": incorrect,
        "efficiency_flag": efficiency,
        "detector_result": json.dumps(detector_summary, separators=(",", ":"), sort_keys=True),
        "final_verdict": record.get("final_verdict"),
        "adjudication_error": record.get("adjudication_error"),
    }


def _collision_route_is_conservative(mesh: dict[str, Any]) -> bool:
    return bool(mesh) and (
        mesh.get("mesh_reliable_for_separation") is not True
        or mesh.get("containment_a_in_b") == "unknown"
        or mesh.get("containment_b_in_a") == "unknown"
    )


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for metric in ("collision", "oob", "support"):
        selected = [row for row in rows if row["metric"] == metric]
        metrics[metric] = {
            "records": len(selected),
            "routed": sum(bool(row["requires_vlm"]) for row in selected),
            "direct_valid": sum(not bool(row["requires_vlm"]) for row in selected),
            "routing_assessments": dict(sorted(Counter(row["routing_assessment"] for row in selected).items())),
            "incorrect_routing": sum(bool(row["incorrect_routing"]) for row in selected),
            "efficiency_flags": sum(bool(row["efficiency_flag"]) for row in selected),
            "gt_status": dict(sorted(Counter(row["gt_status"] for row in selected).items())),
        }
    return {
        "status": "valid_with_support_gt_blocked",
        "routing_records": len(rows),
        "incorrect_routing_records": sum(bool(row["incorrect_routing"]) for row in rows),
        "efficiency_flag_records": sum(bool(row["efficiency_flag"]) for row in rows),
        "metrics": metrics,
        "interpretation": {
            "incorrect_routing": "A route contradicts usable GT or the frozen deterministic routing policy.",
            "efficiency_flag": "A valid event reached the VLM, but the route may still be justified by degraded or non-certifying evidence.",
            "support": "Only policy consistency is audited; routing accuracy is withheld until exact transformed-mesh GT is independently frozen.",
        },
    }


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _write_tsv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    fieldnames: list[str] | None = None,
) -> None:
    if fieldnames is None:
        if not rows:
            raise RuntimeError("cannot infer fields from empty rows")
        fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
