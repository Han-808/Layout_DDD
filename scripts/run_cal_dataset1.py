#!/usr/bin/env python3
"""Run cal_dataset1's local, model-free evaluation arms and audit routing."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from benchmark.evaluator.generic_validity import evaluate_generic_validity  # noqa: E402
from benchmark.evaluator.generic_validity.mesh_geometry import (  # noqa: E402
    load_collision_geometry_manifest,
)
from benchmark.evaluator.spatial_fidelity import (  # noqa: E402
    DEFAULT_COOCCURRENCE_CONFIG,
    DEFAULT_SCALE_CONFIG,
    evaluate_cooccurrence,
    evaluate_scale,
    load_ontology,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=REPO_ROOT / "Support" / "datasets" / "cal_dataset1",
    )
    parser.add_argument("--arm", choices=("proxy", "mesh"), default="proxy")
    parser.add_argument(
        "--geometry-root",
        type=Path,
        default=None,
        help="Root containing <case>/renders/collision_geometry_manifest.json for --arm mesh.",
    )
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    cases_payload = _read_json(root / "cases.json")
    config_path = root / "configs/deterministic_full.json"
    config = _read_json(config_path)
    cases = cases_payload["cases"]
    output_root = root / "evaluation" / args.arm
    errors: list[str] = []
    case_summaries: list[dict[str, Any]] = []
    route_totals = {"collision": 0, "oob": 0, "support": 0}
    target_totals = {
        "confirmed_invalid": 0,
        "confirmed_invalid_routed": 0,
        "reviewed_ambiguous": 0,
        "reviewed_ambiguous_routed": 0,
        "pending_review": 0,
        "pending_review_routed": 0,
    }

    for case in cases:
        if case["evaluation_scope"] != "deterministic_full":
            continue
        case_id = str(case["case_id"])
        fixture = root / str(case["fixture_dir"])
        scene_path = fixture / "generated_scene.json"
        scene = _read_json(scene_path)
        gt = _read_json(fixture / "event_gt.json")
        geometry = None
        geometry_path = None
        if args.arm == "mesh":
            if args.geometry_root is None:
                errors.append(f"{case_id}: --geometry-root is required for mesh arm")
                continue
            geometry_path = _geometry_manifest_path(args.geometry_root.resolve(), case_id)
            if not geometry_path.is_file():
                errors.append(f"{case_id}: missing geometry manifest {geometry_path}")
                continue
            try:
                geometry = load_collision_geometry_manifest(geometry_path)
                geometry["manifest_path"] = str(geometry_path.resolve())
            except Exception as exc:
                errors.append(f"{case_id}: invalid geometry manifest: {type(exc).__name__}: {exc}")
                continue
        try:
            report = evaluate_generic_validity(
                scene,
                config=config,
                collision_geometry=geometry,
                prompt=str(_read_json(fixture / "scene_request.json")["instruction"]),
            )
        except Exception as exc:
            errors.append(f"{case_id}: evaluation failed: {type(exc).__name__}: {exc}")
            continue
        report_path = output_root / case_id / "generic_validity.json"
        _write_json(report_path, report)
        case_errors = _audit_generic_case(case, scene, gt, report)
        errors.extend(f"{case_id}: {message}" for message in case_errors)
        routed = _routed_events(report)
        for metric in route_totals:
            route_totals[metric] += len(routed[metric])
        for event in gt["events"]:
            if event["route_requirement"] != "must_route":
                continue
            semantic = str(event["semantic_label"])
            bucket = {
                "invalid": "confirmed_invalid",
                "ambiguous": "reviewed_ambiguous",
                "pending_review": "pending_review",
            }.get(semantic)
            if bucket is None:
                errors.append(f"{case_id}: must_route event has unsupported semantic label {semantic!r}")
                continue
            target_totals[bucket] += 1
            if str(event["event_id"]) in routed[str(event["metric"])]:
                target_totals[f"{bucket}_routed"] += 1
        case_summaries.append(
            {
                "case_id": case_id,
                "split": case["split"],
                "object_count": len(scene["objects"]),
                "scene_sha256": _sha256(scene_path),
                "geometry_manifest": str(geometry_path) if geometry_path else None,
                "geometry_sha256": _sha256(geometry_path) if geometry_path else None,
                "evaluator_versions": {
                    metric: report["metrics"][metric].get("evaluator_version")
                    for metric in ("collision", "oob", "support")
                },
                "routed_counts": {metric: len(ids) for metric, ids in routed.items()},
                "navigability_score": report["metrics"]["navigability"].get("score"),
                "accessibility_status": report["metrics"]["accessibility"].get("status"),
                "report_path": report_path.relative_to(root).as_posix(),
            }
        )

    spatial_summaries: list[dict[str, Any]] = []
    if args.arm == "proxy":
        ontology_path = root / "ontology/SceneOnto.json"
        ontology = load_ontology(ontology_path)
        for case in cases:
            scope = str(case["evaluation_scope"])
            if scope not in {"scale_only", "cooccurrence_only"}:
                continue
            case_id = str(case["case_id"])
            fixture = root / str(case["fixture_dir"])
            scene = _read_json(fixture / "generated_scene.json")
            prompt = str(_read_json(fixture / "scene_request.json")["instruction"])
            if scope == "scale_only":
                report = evaluate_scale(scene, ontology, deepcopy(DEFAULT_SCALE_CONFIG), prompt=prompt)
                metric = "scale"
                checks = report["checks"]
                routed_checks = [check for check in checks if check.get("route") == "requires_vlm"]
            else:
                report = evaluate_cooccurrence(
                    scene,
                    ontology,
                    deepcopy(DEFAULT_COOCCURRENCE_CONFIG),
                    prompt=prompt,
                )
                metric = "cooccurrence_plausibility"
                checks = report["checks"]
                routed_checks = [check for check in checks if check.get("route") == "requires_vlm"]
            report_path = output_root / case_id / f"{metric}.json"
            _write_json(report_path, report)
            if len(routed_checks) != 1:
                errors.append(f"{case_id}: expected exactly one {metric} requires_vlm check, found {len(routed_checks)}")
            spatial_summaries.append(
                {
                    "case_id": case_id,
                    "metric": metric,
                    "route": routed_checks[0].get("route") if len(routed_checks) == 1 else None,
                    "reason": routed_checks[0].get("reason") if len(routed_checks) == 1 else None,
                    "report_path": report_path.relative_to(root).as_posix(),
                }
            )

    summary = {
        "dataset_id": "cal_dataset1",
        "arm": args.arm,
        "status": "passed" if not errors else "failed",
        "model_calls": 0,
        "gpu_required": False,
        "deterministic_full_case_count": len(case_summaries),
        "spatial_isolated_case_count": len(spatial_summaries),
        "config_path": config_path.relative_to(root).as_posix(),
        "config_sha256": _sha256(config_path),
        "explicitly_excluded_from_full": ["scale", "cooccurrence_plausibility", "functional_grouping"],
        "candidate_route_totals": route_totals,
        "target_routing": target_totals,
        "confirmed_invalid_route_recall": _ratio(
            target_totals["confirmed_invalid_routed"], target_totals["confirmed_invalid"]
        ),
        "reviewed_ambiguous_route_coverage": _ratio(
            target_totals["reviewed_ambiguous_routed"], target_totals["reviewed_ambiguous"]
        ),
        "pending_review_route_coverage": _ratio(
            target_totals["pending_review_routed"], target_totals["pending_review"]
        ),
        "errors": errors,
        "cases": case_summaries,
        "spatial_isolated": spatial_summaries,
        "interpretation": {
            "detector_only_score": None,
            "candidate_is_not_invalid_verdict": True,
            "reviewed_ambiguous_excluded_from_accuracy": True,
            "pending_review_count": target_totals["pending_review"],
            "valid_route_allowed_is_not_automatically_false_positive": True,
        },
    }
    summary_path = output_root / "summary.json"
    _write_json(summary_path, summary)
    print(json.dumps(summary, indent=2))
    if errors:
        raise SystemExit(1)


def _audit_generic_case(
    case: dict[str, Any], scene: dict[str, Any], gt: dict[str, Any], report: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    metrics = report["metrics"]
    n = len(scene["objects"])
    collision = metrics["collision"]
    oob = metrics["oob"]
    support = metrics["support"]
    if len(collision.get("pairs", [])) != n * (n - 1) // 2:
        errors.append("collision pair universe is incomplete")
    if len(oob.get("objects", [])) != n:
        errors.append("OOB object universe is incomplete")
    if len(support.get("objects", [])) != n:
        errors.append("Support object universe is incomplete")
    for metric_name, metric in (("collision", collision), ("oob", oob), ("support", support)):
        if metric.get("status") != "detector_only":
            errors.append(f"{metric_name} status must be detector_only, got {metric.get('status')}")
        if metric.get("score") is not None:
            errors.append(f"{metric_name} detector-only score must be null")
        if metric.get("object_errors"):
            errors.append(f"{metric_name} has object errors: {metric.get('object_errors')}")
    nav = metrics["navigability"]
    nav_score = nav.get("score")
    if nav.get("status") != "checked" or not _score(nav_score):
        errors.append(f"navigability must be checked with [0,1] score, got {nav.get('status')} / {nav_score}")
    accessibility = metrics["accessibility"]
    if accessibility.get("status") != "not_applicable":
        errors.append(f"accessibility should be not_applicable without interactive targets, got {accessibility.get('status')}")
    routed = _routed_events(report)
    for event in gt["events"]:
        if event.get("route_requirement") == "must_route":
            metric = str(event["metric"])
            if str(event["event_id"]) not in routed[metric]:
                errors.append(f"must_route event bypassed: {metric}:{event['event_id']}")
    for record in collision.get("pairs", []):
        _audit_detector_record(record, errors, f"collision:{record.get('object_a')}|{record.get('object_b')}")
    for metric_name, records in (("oob", oob.get("objects", [])), ("support", support.get("objects", []))):
        for record in records:
            _audit_detector_record(record, errors, f"{metric_name}:{record.get('object_id')}")
    if "spatial_fidelity" in report or "scale" in metrics or "cooccurrence_plausibility" in metrics:
        errors.append("deterministic-full report unexpectedly contains Spatial Fidelity")
    if case["prompt_granularity"] == "fine_grained" and case["split"] != "fine_edge":
        errors.append("Fine prompt mode is only allowed in fine_edge split")
    return errors


def _audit_detector_record(record: dict[str, Any], errors: list[str], label: str) -> None:
    if record.get("judge_result") is not None or record.get("adjudication_error") is not None:
        errors.append(f"{label} unexpectedly contains model adjudication")
    if record.get("requires_vlm"):
        if record.get("final_verdict") is not None:
            errors.append(f"{label} candidate fabricated a final verdict")
    else:
        if record.get("final_verdict") != "valid":
            errors.append(f"{label} bypass lacks direct-valid verdict")


def _routed_events(report: dict[str, Any]) -> dict[str, set[str]]:
    metrics = report["metrics"]
    return {
        "collision": {
            "|".join(sorted([str(record["object_a"]), str(record["object_b"])]))
            for record in metrics["collision"].get("pairs", [])
            if record.get("requires_vlm")
        },
        "oob": {
            str(record["object_id"])
            for record in metrics["oob"].get("objects", [])
            if record.get("requires_vlm")
        },
        "support": {
            str(record["object_id"])
            for record in metrics["support"].get("objects", [])
            if record.get("requires_vlm")
        },
    }


def _geometry_manifest_path(root: Path, case_id: str) -> Path:
    candidates = [
        root / case_id / "renders/collision_geometry_manifest.json",
        root / case_id / "collision_geometry_manifest.json",
    ]
    return next((path for path in candidates if path.is_file()), candidates[0])


def _score(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and 0.0 <= float(value) <= 1.0
    )


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else float(numerator) / float(denominator)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
