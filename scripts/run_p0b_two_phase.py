#!/usr/bin/env python3
"""Prepare P0b camera evidence offline, then judge frozen packets concurrently."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any

from benchmark.rendering import BlenderRenderer
from benchmark.rendering.camera_pose import resolve_camera_pose_mode
from benchmark.visual_judge.openai_compatible import build_openai_compatible_vlm_judge
from benchmark.visual_judge.p0b import (
    adjudicate_p0b_event,
    build_p0b_local_evidence_request,
)
from benchmark.visual_judge.render_views import CameraEvidenceProvider
try:
    from scripts.run_p0b_camera_ablation import (
        ARM_CONFIGS,
        _file_sha256,
        _filter_evidence_items,
        _gt_events,
        _json_sha256,
        _judge_identity,
        _load_collision_geometry,
        _read_json,
        _safe_name,
        _source_events,
    )
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from run_p0b_camera_ablation import (  # type: ignore[no-redef]
        ARM_CONFIGS,
        _file_sha256,
        _filter_evidence_items,
        _gt_events,
        _json_sha256,
        _judge_identity,
        _load_collision_geometry,
        _read_json,
        _safe_name,
        _source_events,
    )


SCHEMA_VERSION = "p0b_prepared_evidence_v2"
PREPARATION_RESUME_CONTRACT_VERSION = "p0b_preparation_resume_contract_v1"
JUDGE_RESUME_CONTRACT_VERSION = "p0b_judge_resume_contract_v1"
PREPARED_ARMS = ("fixed_global", "deterministic_metric_local")
SUPPORTED_PACKET_ARMS = (
    "fixed_global",
    "presence_local_raw",
    "presence_local_raw_highlight",
    "presence_global_local_raw",
    "deterministic_metric_local",
    "order_local_first_full",
    "budget_global_first_compact",
    "budget_local_first_compact",
    "vlm_select_from_candidates",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Render one case without a model endpoint.")
    prepare.add_argument("--case-id", required=True)
    prepare.add_argument("--scene", required=True)
    prepare.add_argument("--source-report", required=True)
    prepare.add_argument("--gt", required=True)
    prepare.add_argument("--metric", choices=("collision", "oob", "support"), required=True)
    prepare.add_argument("--blend-file", required=True)
    prepare.add_argument("--overview", action="append", default=[])
    prepare.add_argument("--out-dir", required=True)
    prepare.add_argument("--collision-geometry", default=None)
    prepare.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    prepare.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=False)
    prepare.add_argument("--blender-bin", required=True)
    prepare.add_argument("--blender-timeout-seconds", type=int, default=3600)
    prepare.add_argument("--render-width", type=int, default=512)
    prepare.add_argument("--render-height", type=int, default=512)
    prepare.add_argument(
        "--render-engine",
        choices=("BLENDER_WORKBENCH", "BLENDER_EEVEE_NEXT", "CYCLES"),
        default="CYCLES",
    )
    prepare.add_argument("--cycles-device", choices=("CPU", "CUDA", "OPTIX", "AUTO"), default="CUDA")
    prepare.add_argument("--cycles-samples", type=int, default=8)
    prepare.add_argument("--cycles-denoising", action=argparse.BooleanOptionalAction, default=True)
    prepare.add_argument(
        "--preview-render-engine",
        choices=("BLENDER_WORKBENCH", "BLENDER_EEVEE_NEXT", "CYCLES"),
        default="CYCLES",
    )
    prepare.add_argument("--preview-width", type=int, default=256)
    prepare.add_argument("--preview-height", type=int, default=256)
    prepare.add_argument("--preview-cycles-samples", type=int, default=1)
    prepare.add_argument("--max-views", type=int, default=2)
    prepare.add_argument("--candidate-count", type=int, default=6)
    prepare.add_argument("--collision-overlay", action=argparse.BooleanOptionalAction, default=True)

    judge = subparsers.add_parser("judge", help="Judge prepared evidence without invoking Blender.")
    judge.add_argument("--evidence-root", required=True)
    judge.add_argument("--judge-config", required=True)
    judge.add_argument("--judge-endpoint", default=None)
    judge.add_argument("--judge-model", default=None)
    judge.add_argument("--out-dir", required=True)
    judge.add_argument("--max-workers", type=int, default=4)
    judge.add_argument("--arm", action="append", choices=SUPPORTED_PACKET_ARMS, default=[])
    judge.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    judge.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()
    if args.command == "prepare":
        raise SystemExit(prepare_case(args))
    raise SystemExit(judge_prepared(args))


def prepare_case(args: argparse.Namespace) -> int:
    started = time.time()
    case_id = str(args.case_id)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    scene_path = Path(args.scene).expanduser().resolve()
    report_path = Path(args.source_report).expanduser().resolve()
    gt_path = Path(args.gt).expanduser().resolve()
    blend_path = Path(args.blend_file).expanduser().resolve()
    overview = [str(Path(path).expanduser().resolve()) for path in args.overview]
    for path in [scene_path, report_path, gt_path, blend_path, *(Path(item) for item in overview)]:
        if not path.is_file():
            raise FileNotFoundError(path)

    scene = _read_json(scene_path)
    report = _read_json(report_path)
    gt = _read_json(gt_path)
    gt_events = _gt_events(gt, metrics={str(args.metric)})
    sources = _source_events(report, gt_events)
    file_hashes = {
        "scene": _file_sha256(scene_path),
        "source_report": _file_sha256(report_path),
        "gt": _file_sha256(gt_path),
        "blend_file": _file_sha256(blend_path),
        "overview": [_file_sha256(path) for path in overview],
    }

    renderer = BlenderRenderer(
        blender_bin=args.blender_bin,
        timeout_seconds=args.blender_timeout_seconds,
        width=args.render_width,
        height=args.render_height,
        render_engine=args.render_engine,
        cycles_device=args.cycles_device,
        cycles_samples=args.cycles_samples,
        cycles_denoising=args.cycles_denoising,
        preview_render_engine=args.preview_render_engine,
        preview_width=args.preview_width,
        preview_height=args.preview_height,
        preview_cycles_samples=args.preview_cycles_samples,
    )
    deterministic = ARM_CONFIGS["deterministic_metric_local"]
    collision_geometry = _load_collision_geometry(args.collision_geometry)
    provider = CameraEvidenceProvider(
        renderer=renderer,
        blend_file=blend_path,
        out_dir=out_dir / "deterministic_metric_local" / "camera_evidence",
        mode=str(deterministic["camera_mode"]),
        selector=None,
        metric_modes=dict(deterministic["metric_modes"]),
        max_views=args.max_views,
        max_steps=0,
        candidate_count=args.candidate_count,
        collision_overlay=args.collision_overlay,
        collision_geometry=collision_geometry,
        highlighted_global_pose_policy="legacy_metric",
        candidate_policy="legacy",
    )
    producer_implementation = _source_file_hashes(
        (
            "scripts/run_p0b_two_phase.py",
            "src/benchmark/visual_judge/p0b.py",
        )
    )

    failures: list[dict[str, str]] = []
    prepared = 0
    cached = 0
    for gt_event in gt_events:
        metric = str(gt_event["metric"])
        event_id = str(gt_event["event_id"])
        source = sources[(metric, event_id)]
        local_request = build_p0b_local_evidence_request(
            metric=metric,
            event=source["event"],
            prompt=source["natural_language_prompt"],
            relationships=source["extracted_relationships"],
            scene=scene,
            detector_evidence=source["detector_evidence"],
            object_ids=list(gt_event.get("object_ids") or []),
        )
        common = {
            "schema_version": SCHEMA_VERSION,
            "case_id": case_id,
            "metric": metric,
            "event_id": event_id,
            "gt_label": gt_event["label"],
            "gt_reason_code": gt_event.get("reason_code"),
            "source": deepcopy(source),
            "scene": deepcopy(scene),
            "object_ids": list(local_request["object_ids"]),
            "frozen_event_packet_sha256": _json_sha256(source),
            "frozen_scene_sha256": file_hashes["scene"],
            "frozen_source_report_sha256": file_hashes["source_report"],
            "frozen_gt_sha256": file_hashes["gt"],
            "frozen_blend_sha256": file_hashes["blend_file"],
            "frozen_overview_sha256": file_hashes["overview"],
        }
        input_contract = {
            "case_id": case_id,
            "metric": metric,
            "event_id": event_id,
            "local_request_sha256": _json_sha256(local_request),
            "frozen_event_packet_sha256": common["frozen_event_packet_sha256"],
            "frozen_scene_sha256": common["frozen_scene_sha256"],
            "frozen_source_report_sha256": common["frozen_source_report_sha256"],
            "frozen_gt_sha256": common["frozen_gt_sha256"],
            "frozen_blend_sha256": common["frozen_blend_sha256"],
            "frozen_overview_sha256": common["frozen_overview_sha256"],
        }
        fixed_contract = {
            "schema_version": PREPARATION_RESUME_CONTRACT_VERSION,
            "arm": "fixed_global",
            "inputs": deepcopy(input_contract),
            "producer_implementation": deepcopy(producer_implementation),
            "configuration": {
                "camera_mode": "global_only",
                "evidence_style": "raw",
            },
        }
        fixed_packet = {
            **common,
            "arm": "fixed_global",
            "camera_mode": "global_only",
            "resolved_camera_mode": "global_only",
            "metric_camera_modes": {},
            "evidence_style": "raw",
            "camera_max_steps": 0,
            "pose_selector_enabled": False,
            "overview_render_evidence": overview,
            "local_render_evidence_items": [],
            "frozen_evidence_sha256": _evidence_hashes([], overview),
            "camera_evidence_seconds": 0.0,
            "preparation_contract": fixed_contract,
        }
        fixed_path = _packet_path(out_dir, "fixed_global", metric, event_id)
        if args.resume and _packet_ready(
            fixed_path,
            expected_contract=fixed_contract,
        ):
            cached += 1
        else:
            _write_json(fixed_path, fixed_packet)
            prepared += 1

        deterministic_path = _packet_path(
            out_dir,
            "deterministic_metric_local",
            metric,
            event_id,
        )
        deterministic_contract = {
            "schema_version": PREPARATION_RESUME_CONTRACT_VERSION,
            "arm": "deterministic_metric_local",
            "inputs": deepcopy(input_contract),
            "producer_implementation": deepcopy(producer_implementation),
            "configuration": {
                "camera_mode": str(deterministic["camera_mode"]),
                "metric_camera_modes": deepcopy(deterministic["metric_modes"]),
                "evidence_style": str(deterministic["evidence_style"]),
                "camera_max_steps": 0,
                "candidate_policy": "legacy",
                "max_views": int(args.max_views),
                "candidate_count": int(args.candidate_count),
                "collision_overlay": bool(args.collision_overlay),
                "provider_policy": provider.policy_config,
            },
        }
        if args.resume and _packet_ready(
            deterministic_path,
            expected_contract=deterministic_contract,
        ):
            cached += 1
            continue
        event_started = time.time()
        try:
            raw_items = list(provider(local_request))
            items = _filter_evidence_items(raw_items, str(deterministic["evidence_style"]))
            if not items:
                raise RuntimeError("deterministic provider produced no filtered evidence")
            _validate_evidence_items(items)
            packet = {
                **common,
                "arm": "deterministic_metric_local",
                "camera_mode": str(deterministic["camera_mode"]),
                "resolved_camera_mode": resolve_camera_pose_mode(
                    str(deterministic["camera_mode"]),
                    metric,
                    metric_modes=dict(deterministic["metric_modes"]),
                ),
                "metric_camera_modes": dict(deterministic["metric_modes"]),
                "evidence_style": str(deterministic["evidence_style"]),
                "camera_max_steps": 0,
                "pose_selector_enabled": False,
                "overview_render_evidence": [],
                "local_render_evidence_items": items,
                "frozen_evidence_sha256": _evidence_hashes(items, []),
                "camera_evidence_seconds": time.time() - event_started,
                "preparation_contract": deterministic_contract,
            }
            _write_json(deterministic_path, packet)
            prepared += 1
        except Exception as exc:
            failure = {
                "metric": metric,
                "event_id": event_id,
                "error": f"{type(exc).__name__}: {exc}",
            }
            failures.append(failure)
            _write_json(
                deterministic_path,
                {
                    **common,
                    "arm": "deterministic_metric_local",
                    "camera_mode": str(deterministic["camera_mode"]),
                    "resolved_camera_mode": resolve_camera_pose_mode(
                        str(deterministic["camera_mode"]),
                        metric,
                        metric_modes=dict(deterministic["metric_modes"]),
                    ),
                    "metric_camera_modes": dict(deterministic["metric_modes"]),
                    "evidence_style": str(deterministic["evidence_style"]),
                    "camera_max_steps": 0,
                    "pose_selector_enabled": False,
                    "overview_render_evidence": [],
                    "local_render_evidence_items": [],
                    "frozen_evidence_sha256": [],
                    "camera_evidence_seconds": time.time() - event_started,
                    "preparation_contract": deterministic_contract,
                    "preparation_error": failure["error"],
                },
            )
            if not args.continue_on_error:
                break

    manifest = {
        "schema_version": "p0b_evidence_preparation_manifest_v1",
        "case_id": case_id,
        "metric": args.metric,
        "event_count": len(gt_events),
        "expected_packet_count": len(gt_events) * len(PREPARED_ARMS),
        "prepared_packet_count": prepared,
        "cached_packet_count": cached,
        "failure_count": len(failures),
        "failures": failures,
        "elapsed_seconds": time.time() - started,
        "gpu_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    _write_json(out_dir / "preparation_manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    return 1 if failures else 0


def judge_prepared(args: argparse.Namespace) -> int:
    started = time.time()
    evidence_root = Path(args.evidence_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    judge_config = _read_json(args.judge_config)
    if args.judge_endpoint:
        judge_config["endpoint"] = str(args.judge_endpoint)
    if args.judge_model:
        judge_config["model"] = str(args.judge_model)
    judge_identity = _judge_identity(judge_config)
    judge_contract = _judge_resume_contract(judge_config)
    packet_paths = sorted(evidence_root.glob("*/*/evidence_packets/*.json"))
    if not packet_paths:
        raise FileNotFoundError(f"no prepared evidence packets under {evidence_root}")
    selected_arms = set(args.arm)
    if selected_arms:
        packet_paths = [
            path
            for path in packet_paths
            if str(_read_json(path).get("arm") or "") in selected_arms
        ]
    if not packet_paths:
        raise FileNotFoundError(
            f"no prepared evidence packets for selected arms {sorted(selected_arms)}"
        )
    max_workers = max(1, int(args.max_workers))

    results: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    pending: list[tuple[Path, dict[str, Any], Path]] = []
    for packet_path in packet_paths:
        packet = _read_json(packet_path)
        _validate_packet(packet, packet_path)
        case_id = str(packet["case_id"])
        arm = str(packet["arm"])
        event_path = (
            out_dir
            / case_id
            / arm
            / "events"
            / f"{packet['metric']}__{_safe_name(str(packet['event_id']))}.json"
        )
        if args.resume and _judgement_ready(
            event_path,
            packet_path,
            expected_judge_identity=judge_identity,
            expected_judge_contract=judge_contract,
        ):
            results[(case_id, arm)].append(_read_json(event_path))
        else:
            pending.append((packet_path, packet, event_path))

    def execute(item: tuple[Path, dict[str, Any], Path]) -> tuple[Path, dict[str, Any]]:
        packet_path, packet, event_path = item
        return event_path, _judge_packet(
            packet,
            packet_path,
            judge_config,
            judge_identity,
            judge_contract,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(execute, item): item for item in pending}
        for future in as_completed(futures):
            packet_path, packet, event_path = futures[future]
            try:
                resolved_path, result = future.result()
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                resolved_path = event_path
                result = _failed_judgement_result(
                    packet,
                    packet_path,
                    judge_identity,
                    judge_contract,
                    f"{type(exc).__name__}: {exc}",
                )
            _write_json(resolved_path, result)
            results[(str(packet["case_id"]), str(packet["arm"]))].append(result)
            print(
                f"[{len(results)} groups] {packet['case_id']} {packet['arm']} "
                f"{packet['metric']}:{packet['event_id']} -> {result.get('predicted_label') or 'error'}",
                flush=True,
            )

    for (case_id, arm), group in sorted(results.items()):
        group.sort(key=lambda item: (str(item.get("metric")), str(item.get("event_id"))))
        first = group[0]
        arm_dir = out_dir / case_id / arm
        manifest = {
            "schema_version": "p0b_two_phase_judgement_v1",
            "arm": arm,
            "mode": first["camera_mode"],
            "camera_mode": first["camera_mode"],
            "metric_camera_modes": first.get("metric_camera_modes") or {},
            "camera_max_steps": int(first.get("camera_max_steps") or 0),
            "pose_selector_enabled": bool(first.get("pose_selector_enabled")),
            "pose_selector_model": deepcopy(first.get("pose_selector_model")),
            "final_judge_model": judge_identity,
            "final_judge_contract": deepcopy(judge_contract),
            "evidence_style": first["evidence_style"],
            "canonical_evidence_style": first["evidence_style"],
            "include_overview": bool(first.get("include_overview")),
            "local_presentation": first.get("local_presentation"),
            "global_context": first.get("global_context"),
            "image_order": first.get("image_order"),
            "final_image_budget": first.get("final_image_budget"),
            "max_local_views": first.get("max_local_views"),
            "camera_evidence_dir": (
                str(first.get("camera_evidence_dir"))
                if first.get("camera_evidence_dir")
                else str(evidence_root / case_id / arm / "camera_evidence")
            ),
            "event_count": len(group),
            "elapsed_seconds": sum(float(item.get("elapsed_seconds") or 0.0) for item in group),
            "correct": sum(bool(item.get("match")) for item in group),
            "failed": sum(bool(item.get("error")) for item in group),
            "camera_evidence_seconds": sum(float(item.get("camera_evidence_seconds") or 0.0) for item in group),
            "judge_seconds": sum(float(item.get("judge_seconds") or 0.0) for item in group),
            "estimated_uncached_seconds": sum(float(item.get("estimated_uncached_seconds") or 0.0) for item in group),
            "image_count": sum(int(item.get("image_count") or 0) for item in group),
            "results": group,
        }
        _write_json(arm_dir / "mode_results.json", manifest)

    failure_count = sum(
        bool(item.get("error"))
        for group in results.values()
        for item in group
    )
    manifest = {
        "schema_version": "p0b_two_phase_judge_manifest_v1",
        "evidence_root": str(evidence_root),
        "packet_count": len(packet_paths),
        "newly_judged": len(pending),
        "worker_count": max_workers,
        "final_judge_model": judge_identity,
        "final_judge_contract": deepcopy(judge_contract),
        "failure_count": failure_count,
        "complete": failure_count == 0 and sum(len(group) for group in results.values()) == len(packet_paths),
        "elapsed_seconds": time.time() - started,
    }
    _write_json(out_dir / "judge_manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    return 1 if failure_count else 0


def _judge_packet(
    packet: dict[str, Any],
    packet_path: Path,
    judge_config: dict[str, Any],
    judge_identity: dict[str, Any],
    judge_contract: dict[str, Any],
) -> dict[str, Any]:
    if packet.get("preparation_error"):
        return _failed_judgement_result(
            packet,
            packet_path,
            judge_identity,
            judge_contract,
            f"prepared_evidence_failed: {packet['preparation_error']}",
        )
    source = packet["source"]
    local_items = deepcopy(packet.get("local_render_evidence_items") or [])
    overview = list(packet.get("overview_render_evidence") or [])
    _validate_evidence_items(local_items)
    for path in overview:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    judge = build_openai_compatible_vlm_judge(deepcopy(judge_config))
    started = time.time()
    judgement = adjudicate_p0b_event(
        metric=str(packet["metric"]),
        event=source["event"],
        prompt=source["natural_language_prompt"],
        relationships=source["extracted_relationships"],
        scene=packet["scene"],
        detector_evidence=source["detector_evidence"],
        judge=judge,
        object_ids=list(packet.get("object_ids") or []),
        overview_render_evidence=overview,
        local_view_provider=(lambda _request: deepcopy(local_items)) if local_items else None,
        visual_config_policy="passthrough",
    )
    judge_seconds = time.time() - started
    camera_seconds = float(packet.get("camera_evidence_seconds") or 0.0)
    predicted = str(judgement["verdict"])
    return {
        "arm": packet["arm"],
        "mode": packet["camera_mode"],
        "camera_mode": packet["camera_mode"],
        "resolved_camera_mode": packet["resolved_camera_mode"],
        "metric_camera_modes": packet.get("metric_camera_modes") or {},
        "camera_max_steps": int(packet.get("camera_max_steps") or 0),
        "pose_selector_enabled": bool(packet.get("pose_selector_enabled")),
        "pose_selector_model": deepcopy(packet.get("pose_selector_model")),
        "final_judge_model": judge_identity,
        "final_judge_contract": deepcopy(judge_contract),
        "evidence_style": packet["evidence_style"],
        "canonical_evidence_style": packet["evidence_style"],
        "include_overview": bool(overview),
        "local_presentation": packet.get("local_presentation"),
        "global_context": packet.get("global_context"),
        "image_order": packet.get("image_order"),
        "final_image_budget": packet.get("final_image_budget"),
        "max_local_views": packet.get("max_local_views"),
        "camera_evidence_dir": packet.get("camera_evidence_dir"),
        "metric": packet["metric"],
        "event_id": packet["event_id"],
        "object_ids": list(packet.get("object_ids") or []),
        "gt_label": packet["gt_label"],
        "gt_reason_code": packet.get("gt_reason_code"),
        "predicted_label": predicted,
        "match": predicted == packet["gt_label"],
        "confidence": judgement["confidence"],
        "image_count": len(overview) + len(local_items),
        "evidence_roles": [
            str(item.get("role") or "local") for item in local_items if isinstance(item, dict)
        ] + ["global_raw"] * len(overview),
        "camera_evidence_seconds": camera_seconds,
        "candidate_preview_seconds": float(packet.get("candidate_preview_seconds") or 0.0),
        "selector_seconds": float(packet.get("selector_seconds") or 0.0),
        "final_render_seconds": float(packet.get("final_render_seconds") or 0.0),
        "judge_seconds": judge_seconds,
        "elapsed_seconds": camera_seconds + judge_seconds,
        "phase_b_elapsed_seconds": judge_seconds,
        "estimated_uncached_seconds": camera_seconds + judge_seconds,
        "frozen_event_packet_sha256": packet["frozen_event_packet_sha256"],
        "frozen_scene_sha256": packet["frozen_scene_sha256"],
        "frozen_source_report_sha256": packet["frozen_source_report_sha256"],
        "frozen_gt_sha256": packet["frozen_gt_sha256"],
        "frozen_evidence_sha256": deepcopy(packet["frozen_evidence_sha256"]),
        "prepared_evidence_packet_sha256": _file_sha256(packet_path),
        "judgement": judgement,
    }


def _failed_judgement_result(
    packet: dict[str, Any],
    packet_path: Path,
    judge_identity: dict[str, Any],
    judge_contract: dict[str, Any],
    error: str,
) -> dict[str, Any]:
    camera_seconds = float(packet.get("camera_evidence_seconds") or 0.0)
    return {
        "arm": packet["arm"],
        "mode": packet["camera_mode"],
        "camera_mode": packet["camera_mode"],
        "resolved_camera_mode": packet["resolved_camera_mode"],
        "metric_camera_modes": packet.get("metric_camera_modes") or {},
        "camera_max_steps": int(packet.get("camera_max_steps") or 0),
        "pose_selector_enabled": bool(packet.get("pose_selector_enabled")),
        "pose_selector_model": deepcopy(packet.get("pose_selector_model")),
        "final_judge_model": judge_identity,
        "final_judge_contract": deepcopy(judge_contract),
        "evidence_style": packet["evidence_style"],
        "canonical_evidence_style": packet["evidence_style"],
        "include_overview": bool(packet.get("overview_render_evidence")),
        "local_presentation": packet.get("local_presentation"),
        "global_context": packet.get("global_context"),
        "image_order": packet.get("image_order"),
        "final_image_budget": packet.get("final_image_budget"),
        "max_local_views": packet.get("max_local_views"),
        "camera_evidence_dir": packet.get("camera_evidence_dir"),
        "metric": packet["metric"],
        "event_id": packet["event_id"],
        "object_ids": list(packet.get("object_ids") or []),
        "gt_label": packet["gt_label"],
        "gt_reason_code": packet.get("gt_reason_code"),
        "predicted_label": None,
        "match": False,
        "confidence": None,
        "image_count": len(packet.get("overview_render_evidence") or [])
        + len(packet.get("local_render_evidence_items") or []),
        "camera_evidence_seconds": camera_seconds,
        "candidate_preview_seconds": float(packet.get("candidate_preview_seconds") or 0.0),
        "selector_seconds": float(packet.get("selector_seconds") or 0.0),
        "final_render_seconds": float(packet.get("final_render_seconds") or 0.0),
        "judge_seconds": 0.0,
        "elapsed_seconds": camera_seconds,
        "phase_b_elapsed_seconds": 0.0,
        "estimated_uncached_seconds": camera_seconds,
        "frozen_event_packet_sha256": packet["frozen_event_packet_sha256"],
        "frozen_scene_sha256": packet["frozen_scene_sha256"],
        "frozen_source_report_sha256": packet["frozen_source_report_sha256"],
        "frozen_gt_sha256": packet["frozen_gt_sha256"],
        "frozen_evidence_sha256": deepcopy(packet["frozen_evidence_sha256"]),
        "prepared_evidence_packet_sha256": _file_sha256(packet_path),
        "error": error,
    }


def _source_file_hashes(relative_paths: tuple[str, ...]) -> dict[str, str]:
    repo_root = Path(__file__).resolve().parents[1]
    return {
        relative: _file_sha256(repo_root / relative)
        for relative in relative_paths
    }


def _judge_resume_contract(config: dict[str, Any]) -> dict[str, Any]:
    """Return a secret-free identity for all inputs that affect a judge call."""

    effective_config = {
        "name": config.get("name") or "openai-compatible-vlm-judge",
        "provider": config.get("provider"),
        "endpoint": config.get("endpoint") or config.get("base_url"),
        "model": config.get("model") or config.get("model_id"),
        "api_key_env": config.get("api_key_env"),
        "temperature": float(config.get("temperature", 0.0)),
        "max_tokens": int(config.get("max_tokens", 2048)),
        "context_length": config.get("context_length"),
        "timeout_seconds": int(config.get("timeout_seconds", 300)),
        "response_format_json": bool(config.get("response_format_json", True)),
        "max_retries": int(config.get("max_retries", 1)),
        "retry_backoff_seconds": float(config.get("retry_backoff_seconds", 1.0)),
        "max_tokens_field": str(config.get("max_tokens_field", "max_tokens")),
        "send_temperature": bool(config.get("send_temperature", True)),
        "require_api_key": config.get("require_api_key"),
        "max_images": int(config.get("max_images", 6)),
        "max_context_chars": int(config.get("max_context_chars", 30000)),
    }
    implementation = _source_file_hashes(
        (
            "scripts/run_p0b_two_phase.py",
            "src/benchmark/models/__init__.py",
            "src/benchmark/models/json_response.py",
            "src/benchmark/models/openai_compatible_model.py",
            "src/benchmark/visual_judge/contracts.py",
            "src/benchmark/visual_judge/openai_compatible.py",
            "src/benchmark/visual_judge/p0b.py",
            "src/benchmark/visual_judge/roles.py",
        )
    )
    return {
        "schema_version": JUDGE_RESUME_CONTRACT_VERSION,
        "effective_config": effective_config,
        "implementation": implementation,
        "sha256": _json_sha256(
            {"effective_config": effective_config, "implementation": implementation}
        ),
    }


def _packet_path(out_dir: Path, arm: str, metric: str, event_id: str) -> Path:
    return out_dir / arm / "evidence_packets" / f"{metric}__{_safe_name(event_id)}.json"


def _packet_ready(
    path: Path,
    *,
    expected_contract: dict[str, Any],
) -> bool:
    if not path.is_file():
        return False
    try:
        packet = _read_json(path)
        _validate_packet(packet, path)
    except Exception:
        return False
    return (
        not packet.get("preparation_error")
        and packet.get("preparation_contract") == expected_contract
    )


def _judgement_ready(
    path: Path,
    packet_path: Path,
    *,
    expected_judge_identity: dict[str, Any],
    expected_judge_contract: dict[str, Any],
) -> bool:
    """Reuse only successful results produced from this packet and judge config."""

    if not path.is_file() or not packet_path.is_file():
        return False
    try:
        result = _read_json(path)
        return (
            not result.get("error")
            and result.get("predicted_label") in {"valid", "invalid"}
            and result.get("prepared_evidence_packet_sha256") == _file_sha256(packet_path)
            and result.get("final_judge_model") == expected_judge_identity
            and result.get("final_judge_contract") == expected_judge_contract
        )
    except Exception:
        return False


def _validate_packet(packet: dict[str, Any], path: Path) -> None:
    if packet.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported prepared evidence packet {path}")
    if packet.get("arm") not in SUPPORTED_PACKET_ARMS:
        raise ValueError(f"unsupported prepared arm in {path}: {packet.get('arm')!r}")
    for key in (
        "case_id",
        "metric",
        "event_id",
        "gt_label",
        "source",
        "scene",
        "frozen_event_packet_sha256",
        "frozen_scene_sha256",
        "frozen_source_report_sha256",
        "frozen_gt_sha256",
    ):
        if packet.get(key) in (None, ""):
            raise ValueError(f"prepared packet {path} lacks {key}")
    _validate_evidence_items(packet.get("local_render_evidence_items") or [])
    for image_path in packet.get("overview_render_evidence") or []:
        if not Path(str(image_path)).is_file():
            raise FileNotFoundError(image_path)
    expected_hashes = packet.get("frozen_evidence_sha256")
    if not isinstance(expected_hashes, list):
        raise ValueError(f"prepared packet {path} lacks frozen evidence hashes")
    actual_hashes = _evidence_hashes(
        packet.get("local_render_evidence_items") or [],
        packet.get("overview_render_evidence") or [],
    )
    if actual_hashes != expected_hashes:
        raise ValueError(f"prepared evidence file drift detected in {path}")


def _validate_evidence_items(items: list[Any]) -> None:
    for item in items:
        if not isinstance(item, dict) or not item.get("path"):
            raise ValueError("prepared local evidence items must be objects with path")
        if not Path(str(item["path"])).is_file():
            raise FileNotFoundError(item["path"])


def _evidence_hashes(items: list[Any], overview: list[str]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict) or not item.get("path"):
            raise ValueError("prepared local evidence items must be objects with path")
        path = str(Path(str(item["path"])).expanduser().resolve())
        result.append(
            {
                "path": path,
                "role": str(item.get("role") or "local"),
                "sha256": _file_sha256(path),
            }
        )
    for value in overview:
        path = str(Path(str(value)).expanduser().resolve())
        result.append({"path": path, "role": "global_raw", "sha256": _file_sha256(path)})
    return result


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":
    main()
