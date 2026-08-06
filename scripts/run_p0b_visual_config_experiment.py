#!/usr/bin/env python3
"""Prepare, select, and finalize frozen P0b visual-config evidence.

The commands deliberately separate Blender work from Qwen3-VL-235B calls:

* ``prepare-bank`` composes deterministic presence/order/budget packets and
  renders the six frozen VLM-selector previews for one case;
* ``select`` asks the VLM to choose at most two candidate IDs without allowing
  camera adjustment or invoking Blender;
* ``finalize`` renders only the frozen VLM-selected final views and writes the
  final evidence packet for one case.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any

from benchmark.rendering import BlenderRenderer
from benchmark.rendering.camera_pose import generate_camera_pose_candidates
from benchmark.visual_judge.openai_camera_selector import (
    build_openai_compatible_camera_selector,
)
from benchmark.visual_judge.p0b import build_p0b_local_evidence_request
from benchmark.visual_judge.render_views import (
    CameraEvidenceProvider,
    _require_visible_selector_targets,
)

try:
    from scripts.run_p0b_camera_ablation import (
        ARM_CONFIGS,
        _file_sha256,
        _filter_evidence_items,
        _json_sha256,
        _judge_identity,
        _load_collision_geometry,
        _read_json,
        _safe_name,
    )
    from scripts.run_p0b_two_phase import (
        SCHEMA_VERSION,
        _evidence_hashes,
        _packet_path,
        _validate_evidence_items,
        _validate_packet,
        _write_json,
    )
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from run_p0b_camera_ablation import (  # type: ignore[no-redef]
        ARM_CONFIGS,
        _file_sha256,
        _filter_evidence_items,
        _json_sha256,
        _judge_identity,
        _load_collision_geometry,
        _read_json,
        _safe_name,
    )
    from run_p0b_two_phase import (  # type: ignore[no-redef]
        SCHEMA_VERSION,
        _evidence_hashes,
        _packet_path,
        _validate_evidence_items,
        _validate_packet,
        _write_json,
    )


CANDIDATE_SCHEMA_VERSION = "p0b_vlm_camera_candidate_bank_v1"
SELECTION_SCHEMA_VERSION = "p0b_frozen_vlm_camera_selection_v1"
CANDIDATE_RESUME_CONTRACT_VERSION = "p0b_candidate_bank_resume_contract_v1"
SELECTOR_RESUME_CONTRACT_VERSION = "p0b_selector_resume_contract_v1"
FINALIZE_RESUME_CONTRACT_VERSION = "p0b_finalize_resume_contract_v1"

DETERMINISTIC_VARIANTS: dict[str, dict[str, Any]] = {
    "presence_local_raw": {
        "local_presentation": "raw",
        "global_context": "none",
        "image_order": "local_first",
        "final_image_budget": 2,
        "max_local_views": 2,
    },
    "presence_local_raw_highlight": {
        "local_presentation": "raw_plus_highlight",
        "global_context": "none",
        "image_order": "local_first",
        "final_image_budget": 4,
        "max_local_views": 2,
    },
    "presence_global_local_raw": {
        "local_presentation": "raw",
        "global_context": "metric_highlighted_global",
        "image_order": "global_first",
        "final_image_budget": 3,
        "max_local_views": 2,
    },
    "deterministic_metric_local": {
        "local_presentation": "raw_plus_highlight",
        "global_context": "metric_highlighted_global",
        "image_order": "global_first",
        "final_image_budget": 5,
        "max_local_views": 2,
    },
    "order_local_first_full": {
        "local_presentation": "raw_plus_highlight",
        "global_context": "metric_highlighted_global",
        "image_order": "local_first",
        "final_image_budget": 5,
        "max_local_views": 2,
    },
    "budget_global_first_compact": {
        "local_presentation": "raw_plus_highlight",
        "global_context": "metric_highlighted_global",
        "image_order": "global_first",
        "final_image_budget": 3,
        "max_local_views": 1,
    },
    "budget_local_first_compact": {
        "local_presentation": "raw_plus_highlight",
        "global_context": "metric_highlighted_global",
        "image_order": "local_first",
        "final_image_budget": 3,
        "max_local_views": 1,
    },
}

VISUAL_CONFIG_ARMS = (
    "fixed_global",
    *DETERMINISTIC_VARIANTS,
    "vlm_select_from_candidates",
)

RAW_ROLES = {"metric_local_rgb", "collision_rgb"}
HIGHLIGHT_ROLES = {"metric_local_highlight", "collision_pair_overlay"}
GLOBAL_ROLE = "metric_highlighted_global"


class _UnusedSelector:
    def select_camera_views(self, _request: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("prepare-bank must not invoke the VLM selector")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-bank")
    prepare.add_argument("--case-root", required=True)
    _add_blender_args(prepare)
    prepare.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    prepare.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=False)

    select = subparsers.add_parser("select")
    select.add_argument("--evidence-root", required=True)
    select.add_argument("--judge-config", required=True)
    select.add_argument("--judge-endpoint", default=None)
    select.add_argument("--judge-model", default=None)
    select.add_argument("--max-workers", type=int, default=8)
    select.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    select.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=False)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--case-root", required=True)
    _add_blender_args(finalize)
    finalize.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    finalize.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=False)

    args = parser.parse_args()
    if args.command == "prepare-bank":
        raise SystemExit(prepare_bank(args))
    if args.command == "select":
        raise SystemExit(select_candidates(args))
    raise SystemExit(finalize_case(args))


def _add_blender_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--blend-file", required=True)
    parser.add_argument("--collision-geometry", default=None)
    parser.add_argument("--blender-bin", required=True)
    parser.add_argument("--blender-timeout-seconds", type=int, default=3600)
    parser.add_argument("--render-width", type=int, default=512)
    parser.add_argument("--render-height", type=int, default=512)
    parser.add_argument(
        "--render-engine",
        choices=("BLENDER_WORKBENCH", "BLENDER_EEVEE_NEXT", "CYCLES"),
        default="CYCLES",
    )
    parser.add_argument("--cycles-device", choices=("CPU", "CUDA", "OPTIX", "AUTO"), default="CUDA")
    parser.add_argument("--cycles-samples", type=int, default=8)
    parser.add_argument("--cycles-denoising", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--preview-render-engine",
        choices=("BLENDER_WORKBENCH", "BLENDER_EEVEE_NEXT", "CYCLES"),
        default="CYCLES",
    )
    parser.add_argument("--preview-width", type=int, default=256)
    parser.add_argument("--preview-height", type=int, default=256)
    parser.add_argument("--preview-cycles-samples", type=int, default=1)
    parser.add_argument("--max-views", type=int, default=2)
    parser.add_argument("--candidate-count", type=int, default=6)


def prepare_bank(args: argparse.Namespace) -> int:
    started = time.time()
    case_root = Path(args.case_root).expanduser().resolve()
    blend_file = Path(args.blend_file).expanduser().resolve()
    if not blend_file.is_file():
        raise FileNotFoundError(blend_file)
    base_paths = sorted((case_root / "deterministic_metric_local" / "evidence_packets").glob("*.json"))
    if not base_paths:
        raise FileNotFoundError(f"deterministic evidence packets missing under {case_root}")
    _compose_deterministic_variants(case_root, base_paths)

    renderer = _renderer(args)
    geometry = _load_collision_geometry(args.collision_geometry)
    failures: list[dict[str, str]] = []
    visibility_warnings: list[dict[str, str]] = []
    prepared = 0
    cached = 0
    for base_path in base_paths:
        base = _read_json(base_path)
        event_dir = _candidate_event_dir(case_root, base)
        packet_path = event_dir / "candidate_packet.json"
        provider = CameraEvidenceProvider(
            renderer=renderer,
            blend_file=blend_file,
            out_dir=event_dir / "unused_final_evidence",
            mode="query_cov",
            selector=_UnusedSelector(),
            max_views=int(args.max_views),
            max_steps=0,
            candidate_count=int(args.candidate_count),
            collision_overlay=True,
            collision_geometry=geometry,
            highlighted_global_pose_policy="legacy_metric",
            candidate_policy="legacy",
        )
        preparation_contract = _candidate_preparation_contract(
            args=args,
            base_path=base_path,
            blend_file=blend_file,
            provider=provider,
        )
        if args.resume and _candidate_packet_ready(
            packet_path,
            expected_contract=preparation_contract,
        ):
            cached += 1
            continue
        event_started = time.time()
        try:
            local_request = _local_request_from_packet(base)
            keyed_request = {**local_request, "_resolved_camera_pose_mode": "query_cov"}
            candidates = generate_camera_pose_candidates(
                keyed_request,
                max_candidates=int(args.candidate_count),
                policy="legacy",
            )
            overlay_spec = provider._build_focus_spec(local_request)
            preview_role = "highlighted_focus"
            preview_degradation = None
            preview_visibility_warning = None
            try:
                preview_manifest = provider._render_overlay_views(
                    request=local_request,
                    out_dir=event_dir / "previews",
                    camera_views=candidates,
                    overlay_spec=overlay_spec,
                    preview=True,
                    allow_blank_views=True,
                )
                preview_by_id = _preview_paths(preview_manifest)
                preview_status_by_id = _preview_status_by_id(preview_manifest)
                if not any(status == "ok" for status in preview_status_by_id.values()):
                    raise RuntimeError("all highlighted candidate previews are blank")
            except Exception as exc:
                preview_degradation = f"focus_preview_failed: {type(exc).__name__}: {exc}"
                preview_role = "rgb_fallback"
                preview_manifest = renderer.render_camera_views(
                    blend_file=blend_file,
                    out_dir=event_dir / "previews_rgb_fallback",
                    camera_views=candidates,
                    preview=True,
                    allow_blank_views=True,
                )
                preview_by_id = _preview_paths(preview_manifest)
                preview_status_by_id = _preview_status_by_id(preview_manifest)
                if not any(status == "ok" for status in preview_status_by_id.values()):
                    raise RuntimeError("all candidate preview views are blank")
            if preview_role == "highlighted_focus":
                preview_visibility_warning = _selector_preview_visibility_warning(
                    preview_by_id,
                    overlay_spec,
                )
                if preview_visibility_warning:
                    visibility_warnings.append(
                        {
                            "metric": str(base.get("metric")),
                            "event_id": str(base.get("event_id")),
                            "warning": preview_visibility_warning,
                        }
                    )
            missing = [str(item["id"]) for item in candidates if str(item["id"]) not in preview_by_id]
            if missing:
                raise RuntimeError(f"candidate preview render omitted views: {missing}")
            selector_request = {
                "mode": "query_cov",
                "metric": local_request.get("metric"),
                "event": local_request.get("event"),
                "object_ids": local_request.get("object_ids"),
                "detector_evidence": local_request.get("detector_evidence"),
                "natural_language_prompt": local_request.get("natural_language_prompt"),
                "extracted_relationships": local_request.get("extracted_relationships"),
                "candidates": [
                    {
                        "id": pose["id"],
                        "pose": pose,
                        "image_path": preview_by_id[str(pose["id"])],
                        "render_status": preview_status_by_id.get(str(pose["id"]), "unknown"),
                    }
                    for pose in candidates
                ],
                "max_views": int(args.max_views),
                "step": 0,
                "max_steps": 0,
                "allow_adjustment": False,
                "allowed_actions": [],
                "selection_role": "choose_evidence_views_only_do_not_judge_metric",
                "preview_role": preview_role,
                "preview_degradation": preview_degradation,
                "preview_visibility_warning": preview_visibility_warning,
                "color_legend": overlay_spec.get("legend"),
            }
            preview_hashes = [
                {"id": item["id"], "path": item["image_path"], "sha256": _file_sha256(item["image_path"])}
                for item in selector_request["candidates"]
            ]
            packet = {
                "schema_version": CANDIDATE_SCHEMA_VERSION,
                "case_id": base["case_id"],
                "metric": base["metric"],
                "event_id": base["event_id"],
                "local_request": local_request,
                "selector_request": selector_request,
                "candidate_preview_sha256": preview_hashes,
                "candidate_count": len(candidates),
                "max_views": int(args.max_views),
                "frozen_event_packet_sha256": base["frozen_event_packet_sha256"],
                "frozen_scene_sha256": base["frozen_scene_sha256"],
                "frozen_source_report_sha256": base["frozen_source_report_sha256"],
                "frozen_gt_sha256": base["frozen_gt_sha256"],
                "frozen_blend_sha256": _file_sha256(blend_file),
                "preview_seconds": time.time() - event_started,
                "preview_degradation": preview_degradation,
                "preview_visibility_warning": preview_visibility_warning,
                "blank_candidate_ids": sorted(
                    candidate_id
                    for candidate_id, status in preview_status_by_id.items()
                    if status == "blank"
                ),
                "preparation_contract": preparation_contract,
            }
            _write_json(packet_path, packet)
            prepared += 1
        except Exception as exc:
            failure = {
                "metric": str(base.get("metric")),
                "event_id": str(base.get("event_id")),
                "error": f"{type(exc).__name__}: {exc}",
            }
            failures.append(failure)
            _write_json(
                packet_path,
                {
                    "schema_version": CANDIDATE_SCHEMA_VERSION,
                    "case_id": base.get("case_id"),
                    "metric": base.get("metric"),
                    "event_id": base.get("event_id"),
                    "preparation_contract": preparation_contract,
                    "preparation_error": failure["error"],
                },
            )
            if not args.continue_on_error:
                break
    manifest = {
        "schema_version": "p0b_visual_config_prepare_bank_manifest_v1",
        "case_id": _read_json(base_paths[0])["case_id"],
        "deterministic_variant_arms": list(DETERMINISTIC_VARIANTS),
        "candidate_events": len(base_paths),
        "prepared": prepared,
        "cached": cached,
        "failure_count": len(failures),
        "failures": failures,
        "visibility_warning_count": len(visibility_warnings),
        "visibility_warnings": visibility_warnings,
        "elapsed_seconds": time.time() - started,
    }
    _write_json(case_root / "visual_config_prepare_manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    return 1 if failures else 0


def select_candidates(args: argparse.Namespace) -> int:
    started = time.time()
    evidence_root = Path(args.evidence_root).expanduser().resolve()
    paths = sorted(evidence_root.glob("*/_vlm_candidate_bank/*/candidate_packet.json"))
    if not paths:
        raise FileNotFoundError(f"no VLM camera candidate packets under {evidence_root}")
    config = _read_json(args.judge_config)
    if args.judge_endpoint:
        config["endpoint"] = str(args.judge_endpoint)
    if args.judge_model:
        config["model"] = str(args.judge_model)
    identity = _judge_identity(config)
    selector_contract = _selector_resume_contract(config)
    pending: list[Path] = []
    cached = 0
    for path in paths:
        output = path.with_name("selection_decision.json")
        if args.resume and _selection_ready(
            output,
            path,
            identity,
            selector_contract,
        ):
            cached += 1
        else:
            pending.append(path)

    failures: list[dict[str, str]] = []

    def execute(path: Path) -> tuple[Path, dict[str, Any]]:
        packet = _read_json(path)
        if packet.get("preparation_error"):
            raise RuntimeError(str(packet["preparation_error"]))
        _validate_candidate_packet(packet, path)
        selector = build_openai_compatible_camera_selector(
            deepcopy(config)
        )
        call_started = time.time()
        decision = selector.select_camera_views(
            deepcopy(packet["selector_request"])
        )
        elapsed = time.time() - call_started
        if decision.get("action") is not None:
            raise ValueError("selection-only arm returned a forbidden camera action")
        selected = list(dict.fromkeys(str(value) for value in decision.get("selected_view_ids") or []))
        available = {str(item["id"]) for item in packet["selector_request"]["candidates"]}
        if not selected or len(selected) > int(packet["max_views"]) or any(value not in available for value in selected):
            raise ValueError("selection-only arm returned invalid candidate IDs")
        return path.with_name("selection_decision.json"), {
            "schema_version": SELECTION_SCHEMA_VERSION,
            "case_id": packet["case_id"],
            "metric": packet["metric"],
            "event_id": packet["event_id"],
            "selected_view_ids": selected,
            "camera_adjustment_allowed": False,
            "decision": decision,
            "pose_selector_model": identity,
            "selector_contract": deepcopy(selector_contract),
            "selector_seconds": elapsed,
            "candidate_packet_sha256": _file_sha256(path),
            "candidate_preview_sha256": deepcopy(packet["candidate_preview_sha256"]),
        }

    with ThreadPoolExecutor(max_workers=max(1, int(args.max_workers))) as executor:
        futures = {executor.submit(execute, path): path for path in pending}
        for future in as_completed(futures):
            source_path = futures[future]
            try:
                output, value = future.result()
            except Exception as exc:
                failure = {
                    "candidate_packet": str(source_path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
                failures.append(failure)
                output = source_path.with_name("selection_decision.json")
                value = {
                    "schema_version": SELECTION_SCHEMA_VERSION,
                    "selection_error": failure["error"],
                    "pose_selector_model": identity,
                    "selector_contract": deepcopy(selector_contract),
                    "candidate_packet_sha256": _file_sha256(source_path),
                }
            _write_json(output, value)
            if failures and not args.continue_on_error:
                for item in futures:
                    item.cancel()
                break
    manifest = {
        "schema_version": "p0b_visual_config_selection_manifest_v1",
        "candidate_packet_count": len(paths),
        "selected": len(pending) - len(failures),
        "cached": cached,
        "failure_count": len(failures),
        "failures": failures,
        "pose_selector_model": identity,
        "selector_contract": deepcopy(selector_contract),
        "max_workers": max(1, int(args.max_workers)),
        "elapsed_seconds": time.time() - started,
    }
    _write_json(evidence_root / "selection_manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    return 1 if failures else 0


def finalize_case(args: argparse.Namespace) -> int:
    started = time.time()
    case_root = Path(args.case_root).expanduser().resolve()
    blend_file = Path(args.blend_file).expanduser().resolve()
    candidate_paths = sorted((case_root / "_vlm_candidate_bank").glob("*/candidate_packet.json"))
    if not candidate_paths:
        raise FileNotFoundError(f"candidate bank missing under {case_root}")
    renderer = _renderer(args)
    geometry = _load_collision_geometry(args.collision_geometry)
    arm = ARM_CONFIGS["vlm_select_from_candidates"]
    failures: list[dict[str, str]] = []
    prepared = 0
    cached = 0
    for candidate_path in candidate_paths:
        candidate = _read_json(candidate_path)
        metric = str(candidate.get("metric") or "")
        event_id = str(candidate.get("event_id") or "")
        output = _packet_path(case_root, "vlm_select_from_candidates", metric, event_id)
        decision_path = candidate_path.with_name("selection_decision.json")
        event_started = time.time()
        base_path = _packet_path(case_root, "deterministic_metric_local", metric, event_id)
        base = _read_json(base_path)
        selection_model = None
        finalization_contract = None
        try:
            _validate_candidate_packet(candidate, candidate_path)
            decision = _read_json(decision_path)
            selection_model = deepcopy(decision.get("pose_selector_model"))
            if decision.get("selection_error"):
                raise RuntimeError(str(decision["selection_error"]))
            if decision.get("candidate_packet_sha256") != _file_sha256(candidate_path):
                raise ValueError("camera selection decision does not match candidate packet hash")
            selected_ids = [str(value) for value in decision.get("selected_view_ids") or []]
            decision_sha256 = _file_sha256(decision_path)
            provider = CameraEvidenceProvider(
                renderer=renderer,
                blend_file=blend_file,
                out_dir=(
                    case_root
                    / "vlm_select_from_candidates"
                    / "camera_evidence"
                    / decision_sha256[:16]
                ),
                mode=str(arm["camera_mode"]),
                selector=None,
                metric_modes=dict(arm["metric_modes"]),
                max_views=int(args.max_views),
                max_steps=0,
                candidate_count=int(args.candidate_count),
                collision_overlay=True,
                collision_geometry=geometry,
                frozen_view_ids=selected_ids,
                highlighted_global_pose_policy="legacy_metric",
                candidate_policy="legacy",
            )
            finalization_contract = _finalization_resume_contract(
                args=args,
                base_path=base_path,
                candidate_path=candidate_path,
                decision_path=decision_path,
                blend_file=blend_file,
                provider=provider,
                arm=arm,
            )
            if args.resume and _finalized_packet_ready(
                output,
                expected_contract=finalization_contract,
            ):
                cached += 1
                continue
            raw_items = list(provider(deepcopy(candidate["local_request"])))
            items = _filter_evidence_items(raw_items, str(arm["evidence_style"]))
            _validate_evidence_items(items)
            if not items:
                raise RuntimeError("VLM-selected final evidence is empty")
            render_seconds = time.time() - event_started
            packet = {
                **deepcopy(base),
                "arm": "vlm_select_from_candidates",
                "camera_mode": str(arm["camera_mode"]),
                "resolved_camera_mode": "query_cov",
                "metric_camera_modes": deepcopy(arm["metric_modes"]),
                "evidence_style": str(arm["evidence_style"]),
                "camera_max_steps": 0,
                "pose_selector_enabled": True,
                "pose_selector_model": selection_model,
                "camera_adjustment_allowed": False,
                "selected_view_ids": selected_ids,
                "selector_decision_sha256": decision_sha256,
                "candidate_preview_sha256": deepcopy(candidate["candidate_preview_sha256"]),
                "overview_render_evidence": [],
                "local_render_evidence_items": items,
                "camera_evidence_dir": str(
                    case_root / "vlm_select_from_candidates" / "camera_evidence"
                ),
                "frozen_evidence_sha256": _evidence_hashes(items, []),
                "candidate_preview_seconds": float(candidate.get("preview_seconds") or 0.0),
                "selector_seconds": float(decision.get("selector_seconds") or 0.0),
                "final_render_seconds": render_seconds,
                "camera_evidence_seconds": (
                    float(candidate.get("preview_seconds") or 0.0)
                    + float(decision.get("selector_seconds") or 0.0)
                    + render_seconds
                ),
                "local_presentation": "raw_plus_highlight",
                "global_context": "metric_highlighted_global",
                "image_order": "global_first",
                "final_image_budget": 5,
                "max_local_views": int(args.max_views),
                "finalization_contract": finalization_contract,
            }
            packet.pop("preparation_error", None)
            _write_json(output, packet)
            prepared += 1
        except Exception as exc:
            failure = {"metric": metric, "event_id": event_id, "error": f"{type(exc).__name__}: {exc}"}
            failures.append(failure)
            failed = {
                **deepcopy(base),
                "arm": "vlm_select_from_candidates",
                "camera_mode": str(arm["camera_mode"]),
                "resolved_camera_mode": "query_cov",
                "metric_camera_modes": deepcopy(arm["metric_modes"]),
                "evidence_style": str(arm["evidence_style"]),
                "camera_max_steps": 0,
                "pose_selector_enabled": True,
                "pose_selector_model": selection_model,
                "overview_render_evidence": [],
                "local_render_evidence_items": [],
                "camera_evidence_dir": str(
                    case_root / "vlm_select_from_candidates" / "camera_evidence"
                ),
                "frozen_evidence_sha256": [],
                "finalization_contract": finalization_contract,
                "preparation_error": failure["error"],
            }
            _write_json(output, failed)
            if not args.continue_on_error:
                break
    manifest = {
        "schema_version": "p0b_visual_config_finalize_manifest_v1",
        "candidate_event_count": len(candidate_paths),
        "prepared": prepared,
        "cached": cached,
        "failure_count": len(failures),
        "failures": failures,
        "elapsed_seconds": time.time() - started,
    }
    _write_json(case_root / "vlm_finalize_manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    return 1 if failures else 0


def _compose_deterministic_variants(case_root: Path, base_paths: list[Path]) -> None:
    for base_path in base_paths:
        original = _read_json(base_path)
        source_items = list(original.get("local_render_evidence_items") or [])
        _validate_evidence_items(source_items)
        for arm, factors in DETERMINISTIC_VARIANTS.items():
            items = _compose_items(source_items, **factors)
            if not items:
                raise RuntimeError(f"{arm} produced no evidence for {original['metric']}:{original['event_id']}")
            packet = {
                **deepcopy(original),
                "arm": arm,
                "camera_evidence_dir": str(
                    case_root / "deterministic_metric_local" / "camera_evidence"
                ),
                "overview_render_evidence": [],
                "local_render_evidence_items": items,
                "frozen_evidence_sha256": _evidence_hashes(items, []),
                **deepcopy(factors),
            }
            packet.pop("preparation_error", None)
            _write_json(_packet_path(case_root, arm, str(packet["metric"]), str(packet["event_id"])), packet)

    fixed_paths = sorted((case_root / "fixed_global" / "evidence_packets").glob("*.json"))
    for path in fixed_paths:
        packet = _read_json(path)
        packet.update(
            {
                "camera_evidence_dir": "",
                "local_presentation": "none",
                "global_context": "two_frozen_raw_overviews",
                "image_order": "global_first",
                "final_image_budget": 2,
                "max_local_views": 0,
            }
        )
        _write_json(path, packet)


def _compose_items(
    items: list[dict[str, Any]],
    *,
    local_presentation: str,
    global_context: str,
    image_order: str,
    final_image_budget: int,
    max_local_views: int,
) -> list[dict[str, Any]]:
    global_items = [deepcopy(item) for item in items if str(item.get("role")) == GLOBAL_ROLE][:1]
    raw_items = [deepcopy(item) for item in items if str(item.get("role")) in RAW_ROLES]
    highlight_items = [deepcopy(item) for item in items if str(item.get("role")) in HIGHLIGHT_ROLES]
    highlights_by_view: dict[str, list[dict[str, Any]]] = {}
    for item in highlight_items:
        highlights_by_view.setdefault(_view_key(item), []).append(item)
    local_groups: list[list[dict[str, Any]]] = []
    for index, raw in enumerate(raw_items[:max_local_views]):
        group = [raw]
        if local_presentation == "raw_plus_highlight":
            matches = highlights_by_view.get(_view_key(raw)) or []
            if not matches and index < len(highlight_items):
                matches = [highlight_items[index]]
            group.extend(deepcopy(matches[:1]))
        local_groups.append(group)
    local = [item for group in local_groups for item in group]
    global_context_items = global_items if global_context != "none" else []
    ordered = (
        global_context_items + local
        if image_order == "global_first"
        else local + global_context_items
    )
    if len(ordered) > int(final_image_budget):
        raise ValueError(
            f"evidence bundle has {len(ordered)} images but budget is {final_image_budget}; "
            "bundles may not be truncated mid-pair"
        )
    return ordered


def _view_key(item: dict[str, Any]) -> str:
    if item.get("view_id") is not None:
        return str(item["view_id"])
    pose = item.get("pose") if isinstance(item.get("pose"), dict) else {}
    if pose.get("id") is not None:
        return str(pose["id"])
    return Path(str(item.get("path") or "unknown")).stem


def _local_request_from_packet(packet: dict[str, Any]) -> dict[str, Any]:
    source = packet["source"]
    return build_p0b_local_evidence_request(
        metric=str(packet["metric"]),
        event=source["event"],
        prompt=source["natural_language_prompt"],
        relationships=source["extracted_relationships"],
        scene=packet["scene"],
        detector_evidence=source["detector_evidence"],
        object_ids=list(packet.get("object_ids") or []),
    )


def _renderer(args: argparse.Namespace) -> BlenderRenderer:
    return BlenderRenderer(
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


def _source_file_hashes(relative_paths: tuple[str, ...]) -> dict[str, str]:
    repo_root = Path(__file__).resolve().parents[1]
    return {
        relative: _file_sha256(repo_root / relative)
        for relative in relative_paths
    }


def _candidate_preparation_contract(
    *,
    args: argparse.Namespace,
    base_path: Path,
    blend_file: Path,
    provider: CameraEvidenceProvider,
) -> dict[str, Any]:
    return {
        "schema_version": CANDIDATE_RESUME_CONTRACT_VERSION,
        "base_packet_sha256": _file_sha256(base_path),
        "blend_sha256": _file_sha256(blend_file),
        "collision_geometry_sha256": provider.collision_geometry_sha256,
        "configuration": {
            "candidate_policy": "legacy",
            "max_views": int(args.max_views),
            "candidate_count": int(args.candidate_count),
            "provider_policy": provider.policy_config,
        },
        "producer_implementation": _source_file_hashes(
            (
                "scripts/run_p0b_visual_config_experiment.py",
                "src/benchmark/visual_judge/p0b.py",
            )
        ),
    }


def _selector_resume_contract(config: dict[str, Any]) -> dict[str, Any]:
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
            "scripts/run_p0b_visual_config_experiment.py",
            "src/benchmark/models/__init__.py",
            "src/benchmark/models/json_response.py",
            "src/benchmark/models/openai_compatible_model.py",
            "src/benchmark/visual_judge/contracts.py",
            "src/benchmark/visual_judge/openai_compatible.py",
            "src/benchmark/visual_judge/roles.py",
        )
    )
    return {
        "schema_version": SELECTOR_RESUME_CONTRACT_VERSION,
        "effective_config": effective_config,
        "implementation": implementation,
        "sha256": _json_sha256(
            {"effective_config": effective_config, "implementation": implementation}
        ),
    }


def _finalization_resume_contract(
    *,
    args: argparse.Namespace,
    base_path: Path,
    candidate_path: Path,
    decision_path: Path,
    blend_file: Path,
    provider: CameraEvidenceProvider,
    arm: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": FINALIZE_RESUME_CONTRACT_VERSION,
        "base_packet_sha256": _file_sha256(base_path),
        "candidate_packet_sha256": _file_sha256(candidate_path),
        "selector_decision_sha256": _file_sha256(decision_path),
        "blend_sha256": _file_sha256(blend_file),
        "collision_geometry_sha256": provider.collision_geometry_sha256,
        "configuration": {
            "candidate_policy": "legacy",
            "camera_mode": str(arm["camera_mode"]),
            "metric_camera_modes": deepcopy(arm["metric_modes"]),
            "evidence_style": str(arm["evidence_style"]),
            "max_views": int(args.max_views),
            "candidate_count": int(args.candidate_count),
            "provider_policy": provider.policy_config,
        },
        "producer_implementation": _source_file_hashes(
            (
                "scripts/run_p0b_visual_config_experiment.py",
                "src/benchmark/visual_judge/p0b.py",
            )
        ),
    }


def _candidate_event_dir(case_root: Path, packet: dict[str, Any]) -> Path:
    return (
        case_root
        / "_vlm_candidate_bank"
        / f"{packet['metric']}__{_safe_name(str(packet['event_id']))}"
    )


def _preview_paths(manifest: dict[str, Any]) -> dict[str, str]:
    return {
        str(item["id"]): str(Path(str(item["path"])).expanduser().resolve())
        for item in manifest.get("views") or []
        if isinstance(item, dict) and item.get("id") is not None and item.get("path")
    }


def _preview_status_by_id(manifest: dict[str, Any]) -> dict[str, str]:
    blank_labels = {
        str(value)
        for value in (manifest.get("render_validation") or {}).get("blank_views") or []
    }
    result: dict[str, str] = {}
    for item in manifest.get("views") or []:
        if not isinstance(item, dict) or item.get("id") is None:
            continue
        label = str(item.get("name") or item.get("path"))
        result[str(item["id"])] = "blank" if label in blank_labels else "ok"
    return result


def _selector_preview_visibility_warning(
    preview_by_id: dict[str, str],
    overlay_spec: dict[str, Any],
) -> str | None:
    """Return advisory selector-preview coverage diagnostics without routing events out.

    Candidate previews are evidence offered to the VLM selector, not a deterministic
    proof that a target is absent. Incomplete highlight-pixel coverage is therefore
    recorded for analysis but must not suppress an otherwise valid candidate bank.
    Missing candidate images remain fatal through the separate completeness check.
    """

    try:
        _require_visible_selector_targets(preview_by_id, overlay_spec)
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def _validate_candidate_packet(packet: dict[str, Any], path: Path) -> None:
    if packet.get("schema_version") != CANDIDATE_SCHEMA_VERSION:
        raise ValueError(f"unsupported candidate packet {path}")
    request = packet.get("selector_request")
    if not isinstance(request, dict) or request.get("allow_adjustment") is not False:
        raise ValueError(f"candidate packet {path} violates selection-only contract")
    candidates = request.get("candidates") or []
    if len(candidates) != int(packet.get("candidate_count") or 0):
        raise ValueError(f"candidate count drift in {path}")
    expected = packet.get("candidate_preview_sha256") or []
    actual = [
        {"id": item["id"], "path": item["image_path"], "sha256": _file_sha256(item["image_path"])}
        for item in candidates
    ]
    if actual != expected:
        raise ValueError(f"candidate preview drift detected in {path}")


def _candidate_packet_ready(
    path: Path,
    *,
    expected_contract: dict[str, Any],
) -> bool:
    if not path.is_file():
        return False
    try:
        packet = _read_json(path)
        if packet.get("preparation_error"):
            return False
        _validate_candidate_packet(packet, path)
    except Exception:
        return False
    return packet.get("preparation_contract") == expected_contract


def _selection_ready(
    path: Path,
    candidate_path: Path,
    expected_selector_identity: dict[str, Any],
    expected_selector_contract: dict[str, Any],
) -> bool:
    if not path.is_file():
        return False
    try:
        value = _read_json(path)
        return (
            value.get("schema_version") == SELECTION_SCHEMA_VERSION
            and not value.get("selection_error")
            and value.get("candidate_packet_sha256") == _file_sha256(candidate_path)
            and value.get("pose_selector_model") == expected_selector_identity
            and value.get("selector_contract") == expected_selector_contract
            and bool(value.get("selected_view_ids"))
        )
    except Exception:
        return False


def _finalized_packet_ready(
    path: Path,
    *,
    expected_contract: dict[str, Any],
) -> bool:
    if not path.is_file():
        return False
    try:
        value = _read_json(path)
        _validate_packet(value, path)
        return (
            not value.get("preparation_error")
            and value.get("finalization_contract") == expected_contract
        )
    except Exception:
        return False


if __name__ == "__main__":
    main()
