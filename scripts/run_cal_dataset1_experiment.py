#!/usr/bin/env python3
"""Run model-backed cal_dataset1 experiments from frozen canonical scenes.

This entry point never runs generation, conversion, retrieval, or asset binding.
It consumes the reviewed fixtures directly and keeps the three calibration
tracks explicit:

* deterministic: 31 shared Collision/OOB/Support/Navigability/Accessibility cases;
* spatial: two isolated Scale and two isolated Co-occurrence cases;
* fine_fidelity: the five assistant-converted Fine prompts and frozen references.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from copy import deepcopy
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from benchmark.api.evaluation import run_evaluate  # noqa: E402
from benchmark.evaluator.generic_validity import evaluate_generic_validity  # noqa: E402
from benchmark.evaluator.generic_validity.mesh_geometry import (  # noqa: E402
    validate_collision_geometry_manifest,
)
from benchmark.evaluator.spatial_fidelity import (  # noqa: E402
    DEFAULT_COOCCURRENCE_CONFIG,
    DEFAULT_SCALE_CONFIG,
    evaluate_cooccurrence,
    evaluate_scale,
    load_ontology,
)
from benchmark.reference_annotation import annotation_scoring_gate  # noqa: E402
from benchmark.utils.io import load_yaml  # noqa: E402
from benchmark.visual_judge import build_openai_compatible_vlm_judge  # noqa: E402


TRACKS = ("deterministic", "spatial", "fine_fidelity")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=REPO_ROOT / "Support" / "datasets" / "cal_dataset1",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=REPO_ROOT / "Support" / "artifacts" / "outputs" / "cal_dataset1_experiment",
    )
    parser.add_argument("--judge-config", type=Path, default=None)
    parser.add_argument("--track", action="append", choices=(*TRACKS, "all"), default=[])
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--arm", choices=("mesh", "proxy"), default="mesh")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verify-endpoint", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    root = args.dataset_root.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    cases = _read_json(root / "cases.json")["cases"]
    selected_tracks = _selected_tracks(args.track)
    selected_ids = {str(value) for value in args.case_id}
    selected = [
        case
        for case in cases
        if (not selected_ids or str(case["case_id"]) in selected_ids)
        and _case_selected(case, selected_tracks)
    ]
    unknown = sorted(selected_ids - {str(case["case_id"]) for case in cases})
    if unknown:
        parser.error(f"unknown --case-id values: {unknown}")
    if not selected:
        parser.error("selection produced no experiment cases")

    plan = _build_run_plan(root, selected, selected_tracks, arm=args.arm)
    _write_json(out_root / "run_plan.json", plan)
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return
    if args.judge_config is None:
        parser.error("--judge-config is required unless --plan-only is used")
    judge_config = _read_json(args.judge_config.expanduser().resolve())
    if args.verify_endpoint:
        _verify_endpoint(judge_config)
    judge = build_openai_compatible_vlm_judge(judge_config)
    ontology = load_ontology(root / "ontology/SceneOnto.json")

    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for case in selected:
        case_id = str(case["case_id"])
        tracks = _tracks_for_case(case, selected_tracks)
        for track in tracks:
            report_path = out_root / track / case_id / "evaluation_report.json"
            status_path = report_path.with_name("case_status.json")
            if args.resume and report_path.is_file() and status_path.is_file():
                status = _read_json(status_path)
                if status.get("status") == "completed":
                    results.append(status)
                    continue
            try:
                report, summary = _run_case(
                    root,
                    case,
                    track=track,
                    arm=args.arm,
                    judge=judge,
                    out=report_path,
                    ontology=ontology,
                )
                _write_json(report_path, report)
                status = {
                    "case_id": case_id,
                    "split": case["split"],
                    "track": track,
                    "status": "completed",
                    "report": report_path.relative_to(out_root).as_posix(),
                    **summary,
                }
                _write_json(status_path, status)
                results.append(status)
                print(f"completed {track}:{case_id}")
            except Exception as exc:
                error = {
                    "case_id": case_id,
                    "track": track,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                errors.append(error)
                _write_json(
                    status_path,
                    {**error, "split": case["split"], "status": "failed"},
                )
                print(f"failed {track}:{case_id}: {error['error']}", file=sys.stderr)
                if not args.continue_on_error:
                    break
        if errors and not args.continue_on_error:
            break

    summary = _aggregate_results(root, selected_tracks, args.arm, results, errors, judge_config)
    _write_json(out_root / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    if errors:
        raise SystemExit(1)


def _build_run_plan(
    root: Path,
    cases: list[dict[str, Any]],
    tracks: set[str],
    *,
    arm: str,
) -> dict[str, Any]:
    errors: list[str] = []
    planned: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case["case_id"])
        fixture = root / str(case["fixture_dir"])
        case_tracks = _tracks_for_case(case, tracks)
        for filename in ("generated_scene.json", "scene_request.json", "object_plan.json", "event_gt.json"):
            if not (fixture / filename).is_file():
                errors.append(f"{case_id}: missing {filename}")
        evidence = _render_evidence(root, case_id)
        if any(track in {"deterministic", "fine_fidelity"} for track in case_tracks):
            if len(evidence) != 2:
                errors.append(f"{case_id}: expected two frozen render views, found {len(evidence)}")
        if arm == "mesh" and "deterministic" in case_tracks:
            try:
                _collision_geometry(root, case_id)
            except Exception as exc:
                errors.append(f"{case_id}: mesh geometry is not portable/usable: {exc}")
        if "fine_fidelity" in case_tracks:
            annotation_path = fixture / "reference_annotation.json"
            if not annotation_path.is_file():
                errors.append(f"{case_id}: missing Fine reference annotation")
            else:
                gate = annotation_scoring_gate(_read_json(annotation_path))
                if gate.get("official_scoreable") is not True:
                    errors.append(f"{case_id}: Fine reference is not official-scoreable: {gate}")
        if str(case.get("review_status")) == "pending":
            errors.append(f"{case_id}: human review is still pending")
        planned.append(
            {
                "case_id": case_id,
                "split": case["split"],
                "tracks": case_tracks,
                "prompt_granularity": case["prompt_granularity"],
                "evaluation_scope": case["evaluation_scope"],
                "render_evidence_count": len(evidence),
            }
        )
    plan = {
        "dataset_id": "cal_dataset1",
        "status": "ready" if not errors else "blocked",
        "arm": arm,
        "tracks": sorted(tracks),
        "case_track_run_count": sum(len(item["tracks"]) for item in planned),
        "generator_called": False,
        "runtime_converter_called": False,
        "retrieval_called": False,
        "gpu_required_by_benchmark_code": False,
        "external_vlm_endpoint_required_for_execution": True,
        "errors": errors,
        "cases": planned,
    }
    if errors:
        raise RuntimeError("experiment plan is blocked: " + "; ".join(errors))
    return plan


def _run_case(
    root: Path,
    case: dict[str, Any],
    *,
    track: str,
    arm: str,
    judge: object,
    out: Path,
    ontology: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    case_id = str(case["case_id"])
    fixture = root / str(case["fixture_dir"])
    scene = _read_json(fixture / "generated_scene.json")
    request = _read_json(fixture / "scene_request.json")
    prompt = str(request["instruction"])
    renders = _render_evidence(root, case_id)
    geometry = _collision_geometry(root, case_id) if arm == "mesh" and track == "deterministic" else None

    if track == "deterministic":
        config = _official_generic_config(root / "configs/deterministic_full.json")
        report = evaluate_generic_validity(
            scene,
            config=config,
            collision_geometry=geometry,
            prompt=prompt,
            relationships=[],
            render_evidence=[str(path) for path in renders],
            vlm_judge=judge,
            support_enabled=True,
            p0b_official_mode=True,
        )
        return report, _score_generic_gt(_read_json(fixture / "event_gt.json"), report)

    if track == "spatial":
        scope = str(case["evaluation_scope"])
        if scope == "scale_only":
            report = evaluate_scale(
                scene,
                ontology,
                deepcopy(DEFAULT_SCALE_CONFIG),
                prompt=prompt,
                render_evidence=[str(path) for path in renders],
                vlm_judge=judge,
            )
        elif scope == "cooccurrence_only":
            report = evaluate_cooccurrence(
                scene,
                ontology,
                deepcopy(DEFAULT_COOCCURRENCE_CONFIG),
                prompt=prompt,
                render_evidence=[str(path) for path in renders],
                vlm_judge=judge,
            )
        else:
            raise ValueError(f"spatial track cannot run scope {scope!r}")
        return report, _score_spatial_gt(_read_json(fixture / "event_gt.json"), report)

    if track == "fine_fidelity":
        profile = load_yaml(REPO_ROOT / "configs/evaluation/metric_profile_draft_v1.yaml", default={})
        profile["weights"] = {
            "prompt_fidelity": 1.0,
            "spatial_fidelity": 1.0,
            "structural_validity": 0.0,
            "visual_quality": 0.0,
        }
        report = run_evaluate(
            scene=scene,
            out=out,
            eval_oor=True,
            eval_oar=True,
            eval_generic_validity=False,
            scene_request=request,
            object_plan=_read_json(fixture / "object_plan.json"),
            reference_annotation=_read_json(fixture / "reference_annotation.json"),
            collision_geometry=_collision_geometry(root, case_id) if arm == "mesh" else None,
            render_evidence=[str(path) for path in renders],
            vlm_judge=judge,
            evaluation_profile=profile,
        )
        category = report.get("category_reports", {}).get("prompt_fidelity", {})
        return report, {
            "score": category.get("score"),
            "score_status": category.get("status"),
            "model_call_count": _fine_model_calls(report),
            "official_scoreable_reference": True,
        }
    raise ValueError(f"unknown track {track!r}")


def _official_generic_config(path: Path) -> dict[str, Any]:
    config = _read_json(path)
    for metric in ("collision", "oob", "support"):
        config.setdefault(metric, {})["detector_only"] = False
        config[metric]["official_mode"] = True
    return config


def _collision_geometry(root: Path, case_id: str) -> dict[str, Any]:
    manifest_path = root / f"evaluation/mesh_geometry/{case_id}/renders/collision_geometry_manifest.json"
    raw = _read_json(manifest_path)
    geometry_dir = manifest_path.parent / "collision_geometry"
    for object_id, entry in raw.get("objects", {}).items():
        if not isinstance(entry, dict) or entry.get("representation") != "triangle_mesh":
            continue
        original = Path(str(entry.get("geometry_path") or f"{object_id}.ply"))
        entry["geometry_path"] = str((geometry_dir / original.name).resolve())
    validate_collision_geometry_manifest(raw)
    raw["manifest_path"] = str(manifest_path)
    return raw


def _render_evidence(root: Path, case_id: str) -> list[Path]:
    render_dir = root / f"evaluation/mesh_geometry/{case_id}/renders"
    return [
        path
        for path in (render_dir / "standardized_top.png", render_dir / "standardized_perspective.png")
        if path.is_file()
    ]


def _score_generic_gt(gt: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    outcomes: dict[tuple[str, str], str | None] = {}
    metrics = report["metrics"]
    for record in metrics["collision"].get("pairs", []):
        event_id = "|".join(sorted([str(record["object_a"]), str(record["object_b"])]))
        outcomes[("collision", event_id)] = record.get("final_verdict")
    for metric in ("oob", "support"):
        for record in metrics[metric].get("objects", []):
            outcomes[(metric, str(record["object_id"]))] = record.get("final_verdict")
    return _score_events(gt, outcomes, model_calls=_generic_model_calls(report))


def _score_spatial_gt(gt: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    event = gt["events"][0]
    checks = [check for check in report.get("checks", []) if isinstance(check, dict)]
    verdict = checks[0].get("final_verdict") if len(checks) == 1 else None
    outcomes = {(str(event["metric"]), str(event["event_id"])): verdict}
    model_calls = sum(check.get("route") == "vlm_adjudicated" for check in checks)
    return _score_events(gt, outcomes, model_calls=model_calls)


def _score_events(
    gt: dict[str, Any],
    outcomes: dict[tuple[str, str], str | None],
    *,
    model_calls: int,
) -> dict[str, Any]:
    eligible = correct = unresolved = ambiguous = 0
    false_invalid = false_valid = 0
    for event in gt.get("events", []):
        semantic = str(event.get("semantic_label"))
        if semantic == "ambiguous":
            ambiguous += 1
            continue
        if semantic not in {"valid", "invalid"}:
            continue
        eligible += 1
        verdict = outcomes.get((str(event["metric"]), str(event["event_id"])))
        if verdict not in {"valid", "invalid"}:
            unresolved += 1
        elif verdict == semantic:
            correct += 1
        elif semantic == "valid":
            false_invalid += 1
        else:
            false_valid += 1
    return {
        "score": (correct / float(eligible)) if eligible else None,
        "eligible_gt_event_count": eligible,
        "correct_gt_event_count": correct,
        "unresolved_gt_event_count": unresolved,
        "ambiguous_gt_event_count": ambiguous,
        "false_invalid_count": false_invalid,
        "false_valid_count": false_valid,
        "model_call_count": int(model_calls),
    }


def _generic_model_calls(report: dict[str, Any]) -> int:
    metrics = report.get("metrics", {})
    records = list(metrics.get("collision", {}).get("pairs", []))
    records.extend(metrics.get("oob", {}).get("objects", []))
    records.extend(metrics.get("support", {}).get("objects", []))
    return sum(record.get("route") == "vlm_adjudicated" for record in records if isinstance(record, dict))


def _fine_model_calls(report: dict[str, Any]) -> int:
    reports = report.get("reports", {})
    checks = []
    for family in ("oor", "oar"):
        checks.extend(reports.get(family, {}).get("checks", []))
    return sum(
        check.get("route") == "vlm_adjudicated"
        for check in checks
        if isinstance(check, dict)
    )


def _aggregate_results(
    root: Path,
    tracks: set[str],
    arm: str,
    results: list[dict[str, Any]],
    errors: list[dict[str, str]],
    judge_config: dict[str, Any],
) -> dict[str, Any]:
    eligible = sum(int(item.get("eligible_gt_event_count") or 0) for item in results)
    correct = sum(int(item.get("correct_gt_event_count") or 0) for item in results)
    unresolved = sum(int(item.get("unresolved_gt_event_count") or 0) for item in results)
    return {
        "dataset_id": "cal_dataset1",
        "status": "completed" if not errors else "completed_with_errors",
        "arm": arm,
        "tracks": sorted(tracks),
        "completed_case_track_count": len(results),
        "failed_case_track_count": len(errors),
        "model_call_count": sum(int(item.get("model_call_count") or 0) for item in results),
        "eligible_gt_event_count": eligible,
        "correct_gt_event_count": correct,
        "unresolved_gt_event_count": unresolved,
        "micro_accuracy": (correct / float(eligible)) if eligible else None,
        "ambiguous_events_excluded_from_accuracy": True,
        "judge": {
            "endpoint": str(judge_config.get("endpoint") or judge_config.get("base_url") or ""),
            "model": str(judge_config.get("model") or judge_config.get("model_id") or ""),
            "api_key_env": str(judge_config.get("api_key_env") or "") or None,
        },
        "ontology": {
            "path": str(root / "ontology/SceneOnto.json"),
            "sha256": (root / "dataset_manifest.json").is_file()
            and (_read_json(root / "dataset_manifest.json").get("ontology") or {}).get("sha256"),
        },
        "results": results,
        "errors": errors,
    }


def _verify_endpoint(config: dict[str, Any]) -> None:
    endpoint = str(config.get("endpoint") or config.get("base_url") or "").rstrip("/")
    model = str(config.get("model") or config.get("model_id") or "").strip()
    if not endpoint or not model:
        raise ValueError("judge config requires exact endpoint and model")
    request = urllib.request.Request(f"{endpoint}/models")
    key_env = str(config.get("api_key_env") or "").strip()
    if key_env:
        value = os.environ.get(key_env)
        if not value:
            raise RuntimeError(f"judge credential environment variable {key_env!r} is not set")
        request.add_header("Authorization", f"Bearer {value}")
    with urllib.request.urlopen(request, timeout=float(config.get("timeout_seconds") or 30)) as response:
        payload = json.loads(response.read().decode("utf-8"))
    served = {str(item.get("id")) for item in payload.get("data", []) if isinstance(item, dict)}
    if model not in served:
        raise RuntimeError(f"requested judge model {model!r} is not served; live IDs: {sorted(served)}")


def _selected_tracks(values: list[str]) -> set[str]:
    if not values:
        return {"deterministic"}
    return set(TRACKS) if "all" in values else {str(value) for value in values}


def _case_selected(case: dict[str, Any], tracks: set[str]) -> bool:
    return bool(_tracks_for_case(case, tracks))


def _tracks_for_case(case: dict[str, Any], tracks: set[str]) -> list[str]:
    result: list[str] = []
    if "deterministic" in tracks and case.get("evaluation_scope") == "deterministic_full":
        result.append("deterministic")
    if "spatial" in tracks and case.get("evaluation_scope") in {"scale_only", "cooccurrence_only"}:
        result.append("spatial")
    if "fine_fidelity" in tracks and case.get("split") == "fine_edge":
        result.append("fine_fidelity")
    return result


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
