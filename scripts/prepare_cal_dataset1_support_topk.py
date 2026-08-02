#!/usr/bin/env python3
"""Prepare a nested Support Top-K visual-evidence ablation.

The input is the already materialized ``exp1_1`` camera audit.  Existing
high-resolution Top-1/Top-2 local highlight images and the existing highlighted
global context image are reused byte-for-byte.  Only the third deterministic
local pose is rendered when it is not already present.

The three frozen arms isolate one variable:

* support_top1: 1 local highlight + the same global context (budget 2)
* support_top2: 2 local highlights + the same global context (budget 3)
* support_top3: 3 local highlights + the same global context (budget 4)

Raw/highlight duplication, global presence, image order, scene, detector
evidence, candidate bank, selector and judge are otherwise frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from benchmark.rendering import BlenderRenderer, CYCLES_DEVICES, RENDER_ENGINES  # noqa: E402


SOURCE_COMPARISON_SCHEMA = "cal_dataset1_camera_evidence_comparison_v2"
OUTPUT_SCHEMA = "cal_dataset1_support_topk_comparison_v1"
PLAN_SCHEMA = "cal_dataset1_support_topk_prepare_plan_v1"
RUN_SCHEMA = "cal_dataset1_support_topk_prepare_run_v1"
ARMS = ("support_top1", "support_top2", "support_top3")


def main() -> None:
    args = _parse_args()
    source_roots = [Path(value).expanduser().resolve() for value in args.source_root]
    out_dir = Path(args.out_dir).expanduser().resolve()
    sources = _discover_support_sources(source_roots)
    plan = _plan(args, source_roots, out_dir, sources)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "experiment_plan.json", plan)
    if args.plan_only:
        print(json.dumps(plan["counts"], indent=2))
        return

    renderer = BlenderRenderer(
        blender_bin=Path(args.blender_bin),
        timeout_seconds=args.blender_timeout_seconds,
        width=args.render_width,
        height=args.render_height,
        render_engine=args.render_engine,
        cycles_device=args.cycles_device,
        cycles_samples=args.cycles_samples,
        cycles_denoising=args.cycles_denoising,
        preview_render_engine=args.render_engine,
        preview_width=256,
        preview_height=256,
        preview_cycles_samples=1,
        require_asset_mesh=False,
    )
    started = time.time()
    completed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, source in enumerate(sources, start=1):
        label = f"{source['case_id']} support:{source['event_id']}"
        print(f"[{index}/{len(sources)}] {label}", flush=True)
        try:
            result = _prepare_one(
                source,
                out_dir=out_dir,
                renderer=renderer,
                resume=args.resume,
            )
            completed.append(result)
            print(f"  {result['status']}: {', '.join(result['selected_top3'])}", flush=True)
        except Exception as exc:
            failure = {
                "case_id": source["case_id"],
                "event_id": source["event_id"],
                "error": f"{type(exc).__name__}: {exc}",
            }
            failures.append(failure)
            print(f"  failed: {failure['error']}", file=sys.stderr, flush=True)
            if not args.continue_on_error:
                break

    run = {
        **plan,
        "schema_version": RUN_SCHEMA,
        "elapsed_seconds": time.time() - started,
        "completed_event_count": len(completed),
        "failed_event_count": len(failures),
        "new_render_count": sum(int(item["new_render_count"]) for item in completed),
        "cached_event_count": sum(item["status"] == "cached" for item in completed),
        "completed": completed,
        "failures": failures,
    }
    _write_json(out_dir / "run_manifest.json", run)
    print(json.dumps({
        "completed_event_count": run["completed_event_count"],
        "failed_event_count": run["failed_event_count"],
        "new_render_count": run["new_render_count"],
        "evidence_root": str(out_dir),
    }, indent=2))
    if failures:
        raise SystemExit(1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        action="append",
        default=[],
        help="Existing exp1_1-style evidence root; repeat for invalid and fine-edge sets.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(
            PROJECT_ROOT
            / "Support"
            / "artifacts"
            / "outputs"
            / "exp2_support_topk_evidence"
        ),
    )
    parser.add_argument(
        "--blender-bin",
        default="/Applications/Blender.app/Contents/MacOS/Blender",
    )
    parser.add_argument("--render-width", type=int, default=512)
    parser.add_argument("--render-height", type=int, default=512)
    parser.add_argument(
        "--render-engine",
        choices=RENDER_ENGINES,
        default="BLENDER_WORKBENCH",
    )
    parser.add_argument("--cycles-device", choices=CYCLES_DEVICES, default="CPU")
    parser.add_argument("--cycles-samples", type=int, default=8)
    parser.add_argument(
        "--cycles-denoising",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--blender-timeout-seconds", type=int, default=1800)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if not args.source_root:
        args.source_root = [
            str(
                PROJECT_ROOT
                / "Support"
                / "artifacts"
                / "outputs"
                / "exp1_1"
            ),
            str(
                PROJECT_ROOT
                / "Support"
                / "artifacts"
                / "outputs"
                / "exp1_1_fine_edge"
            ),
        ]
    return args


def _discover_support_sources(source_roots: list[Path]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for root in source_roots:
        if not root.is_dir():
            raise FileNotFoundError(root)
        for path in sorted(root.glob("cases/*/events/support__*/comparison_manifest.json")):
            comparison = _read_json(path)
            if comparison.get("schema_version") != SOURCE_COMPARISON_SCHEMA:
                raise ValueError(f"unsupported source comparison schema: {path}")
            if comparison.get("metric") != "support":
                continue
            key = (str(comparison.get("case_id")), str(comparison.get("event_id")))
            if key in seen:
                raise ValueError(f"duplicate Support event across source roots: {key}")
            seen.add(key)
            local = (comparison.get("arms") or {}).get("metric_local_highlight")
            if not isinstance(local, dict) or not local.get("camera_evidence_manifest"):
                raise ValueError(f"source lacks metric-local manifest: {path}")
            camera_manifest = Path(
                str(local["camera_evidence_manifest"])
            ).expanduser().resolve()
            event_dir = camera_manifest.parent
            case_id = key[0]
            sources.append({
                "case_id": case_id,
                "event_id": key[1],
                "source_root": root,
                "comparison_path": path.resolve(),
                "comparison": comparison,
                "camera_manifest_path": camera_manifest,
                "event_dir": event_dir,
                "pose_candidates_path": event_dir / "pose_candidates.json",
                "overlay_spec_path": event_dir / "focus_overlay_spec.json",
                "blend_file": root / "cases" / case_id / "scene" / "scene.blend",
            })
    if not sources:
        raise ValueError("no Support comparison manifests found")
    sources.sort(key=lambda item: (
        str(item["comparison"].get("severity_class")),
        str(item["case_id"]),
        str(item["event_id"]),
    ))
    return sources


def _plan(
    args: argparse.Namespace,
    source_roots: list[Path],
    out_dir: Path,
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": PLAN_SCHEMA,
        "experiment_id": "exp2_support_topk",
        "experiment_type": "support_nested_local_view_count_ablation",
        "source_roots": [str(path) for path in source_roots],
        "out_dir": str(out_dir),
        "controlled_variable": "number_of_deterministically_ranked_local_highlight_views",
        "arms": {
            "support_top1": {"local_views": 1, "fixed_global_views": 1, "image_budget": 2},
            "support_top2": {"local_views": 2, "fixed_global_views": 1, "image_budget": 3},
            "support_top3": {"local_views": 3, "fixed_global_views": 1, "image_budget": 4},
        },
        "frozen": {
            "scene": True,
            "prompt": True,
            "detector_evidence": True,
            "candidate_bank": True,
            "selector": "support_contact_plane_visibility_rank_v1",
            "presentation": "highlight_only",
            "global_context": "same_existing_metric_highlighted_global",
            "image_order": "local_first_then_global",
            "ground_truth_sent_to_judge": False,
            "existing_top1_top2_images_reused_by_hash": True,
            "only_missing_third_local_view_is_rendered": True,
        },
        "render": {
            "engine": args.render_engine,
            "size": [args.render_width, args.render_height],
            "cycles_device": args.cycles_device,
            "cycles_samples": args.cycles_samples,
        },
        "counts": {
            "events": len(sources),
            "invalid_events": sum(
                item["comparison"].get("semantic_label") == "invalid"
                for item in sources
            ),
            "ambiguous_events": sum(
                item["comparison"].get("semantic_label") == "ambiguous"
                for item in sources
            ),
            "judge_calls_after_preparation": len(sources) * len(ARMS),
        },
    }


def _prepare_one(
    source: dict[str, Any],
    *,
    out_dir: Path,
    renderer: BlenderRenderer,
    resume: bool,
) -> dict[str, Any]:
    for key in (
        "comparison_path",
        "camera_manifest_path",
        "pose_candidates_path",
        "overlay_spec_path",
        "blend_file",
    ):
        path = Path(source[key])
        if not path.is_file():
            raise FileNotFoundError(path)
    comparison = deepcopy(source["comparison"])
    camera_manifest = _read_json(source["camera_manifest_path"])
    candidates_value = json.loads(
        Path(source["pose_candidates_path"]).read_text(encoding="utf-8")
    )
    if not isinstance(candidates_value, list) or len(candidates_value) < 3:
        raise ValueError("Support TopK requires at least three pose candidates")
    candidates = [item for item in candidates_value if isinstance(item, dict)]
    overlay_spec = _read_json(source["overlay_spec_path"])
    selected_top3, ranking = _topk_selection(
        candidates,
        camera_manifest,
        overlay_spec,
    )
    selected_ids = [str(item["id"]) for item in selected_top3]
    existing_selected = [
        str(value)
        for value in (camera_manifest.get("selection") or {}).get(
            "selected_view_ids", []
        )
    ]
    if selected_ids[:2] != existing_selected[:2]:
        raise RuntimeError(
            "current Top-3 ranking is not nested with the frozen Top-2 selection: "
            f"top3={selected_ids}, frozen_top2={existing_selected}"
        )

    event_out = (
        out_dir
        / "cases"
        / str(source["case_id"])
        / "events"
        / f"support__{_safe_name(str(source['event_id']))}"
    )
    output_manifest = event_out / "comparison_manifest.json"
    source_contract = _source_contract(source, selected_ids)
    if resume and _output_ready(output_manifest, source_contract):
        return {
            "case_id": source["case_id"],
            "event_id": source["event_id"],
            "status": "cached",
            "new_render_count": 0,
            "selected_top3": selected_ids,
            "comparison_manifest": str(output_manifest),
        }

    camera_items = [
        deepcopy(item)
        for item in camera_manifest.get("render_evidence_items") or []
        if isinstance(item, dict)
    ]
    global_items = [
        item for item in camera_items
        if item.get("role") == "metric_highlighted_global"
    ]
    if len(global_items) != 1:
        raise RuntimeError(
            f"expected exactly one frozen highlighted global item, found {len(global_items)}"
        )
    local_by_id = {
        str(item.get("view_id")): item
        for item in camera_items
        if item.get("role") == "metric_local_highlight"
    }
    new_render_count = 0
    missing = [
        pose for pose in selected_top3
        if str(pose.get("id")) not in local_by_id
    ]
    if missing:
        bundle = renderer.render_focus_evidence_bundle(
            blend_file=source["blend_file"],
            out_dir=event_out / "rendered_missing_local",
            local_camera_views=missing,
            global_camera_views=[],
            overlay_spec=overlay_spec,
        )
        rendered = {
            str(item.get("id")): item
            for item in bundle.get("overlay_views") or []
            if isinstance(item, dict)
        }
        for pose in missing:
            view_id = str(pose["id"])
            item = rendered.get(view_id)
            if item is None:
                raise RuntimeError(f"renderer omitted missing Support view {view_id}")
            local_by_id[view_id] = {
                "path": str(Path(str(item["path"])).resolve()),
                "role": "metric_local_highlight",
                "view_id": view_id,
                "metric": "support",
                "target_ids": [
                    str(target.get("id"))
                    for target in overlay_spec.get("targets") or []
                    if isinstance(target, dict) and target.get("id") is not None
                ],
                "color_legend": deepcopy(overlay_spec.get("legend")),
                "representation_level": overlay_spec.get("representation_level"),
                "pose": deepcopy(pose),
            }
            new_render_count += 1

    normalized_global = _normalize_item(global_items[0])
    normalized_local = {
        view_id: _normalize_item(local_by_id[view_id])
        for view_id in selected_ids
    }
    arms: dict[str, Any] = {}
    for k, arm in enumerate(ARMS, start=1):
        items = [deepcopy(normalized_local[view_id]) for view_id in selected_ids[:k]]
        items.append(deepcopy(normalized_global))
        arms[arm] = {
            "local_view_count": k,
            "global_view_count": 1,
            "image_count": len(items),
            "selected_local_view_ids": selected_ids[:k],
            "items": items,
        }
    payload = {
        "schema_version": OUTPUT_SCHEMA,
        "case_id": comparison["case_id"],
        "split": comparison.get("split"),
        "metric": "support",
        "event_id": comparison["event_id"],
        "object_ids": deepcopy(comparison.get("object_ids") or []),
        "semantic_label": comparison.get("semantic_label"),
        "severity_class": comparison.get("severity_class"),
        "gt_basis": comparison.get("gt_basis"),
        "presentation": "highlight_only",
        "global_context_policy": "same_existing_metric_highlighted_global",
        "image_order": "local_first_then_global",
        "controlled_variable": "local_view_count",
        "arms": arms,
        "selection": {
            "selector": ranking.get("selector"),
            "selected_top3_view_ids": selected_ids,
            "ranking": ranking,
            "nested_prefix_contract": True,
        },
        "source_contract": source_contract,
        "source_sha256": deepcopy(comparison.get("source_sha256") or {}),
        "source_paths": {
            "comparison_manifest": str(Path(source["comparison_path"]).resolve()),
            "camera_evidence_manifest": str(
                Path(source["camera_manifest_path"]).resolve()
            ),
            "pose_candidates": str(Path(source["pose_candidates_path"]).resolve()),
            "focus_overlay_spec": str(Path(source["overlay_spec_path"]).resolve()),
            "blend_file": str(Path(source["blend_file"]).resolve()),
        },
        "ground_truth_visibility": "scoring_only_not_sent_to_judge",
    }
    event_out.mkdir(parents=True, exist_ok=True)
    _write_json(output_manifest, payload)
    return {
        "case_id": source["case_id"],
        "event_id": source["event_id"],
        "status": "prepared",
        "new_render_count": new_render_count,
        "selected_top3": selected_ids,
        "comparison_manifest": str(output_manifest),
    }


def _topk_selection(
    candidates: list[dict[str, Any]],
    camera_manifest: dict[str, Any],
    overlay_spec: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selection = camera_manifest.get("selection")
    ranking = (
        selection.get("ranking")
        if isinstance(selection, dict) and isinstance(selection.get("ranking"), dict)
        else {}
    )
    scored: list[dict[str, Any]] = []
    for entry in ranking.get("ranked") or []:
        if not isinstance(entry, dict) or entry.get("id") is None:
            continue
        scored.append(deepcopy(entry))
    if len(scored) != len(candidates):
        raise RuntimeError(
            "frozen Support ranking does not cover every candidate; "
            "cannot construct a controlled nested TopK arm"
        )
    by_id = {str(candidate.get("id")): candidate for candidate in candidates}
    available = [entry for entry in scored if entry.get("usable")]
    selected_ids: list[str] = []
    while available and len(selected_ids) < 3:
        def selection_key(entry: dict[str, Any]) -> tuple[float, float, str]:
            diversity = _minimum_angular_diversity(
                by_id[str(entry["id"])],
                [by_id[view_id] for view_id in selected_ids],
            )
            return (
                float(entry.get("base_score") or 0.0) + 0.60 * diversity,
                diversity,
                str(entry["id"]),
            )

        chosen = max(available, key=selection_key)
        selected_ids.append(str(chosen["id"]))
        available = [
            entry for entry in available
            if str(entry["id"]) != str(chosen["id"])
        ]
    fallback_reason: str | None = None
    if len(selected_ids) < 3:
        fallback_reason = (
            "no_candidate_exposed_subject_and_gap"
            if not selected_ids
            else "insufficient_gap_visible_candidates"
        )
        remaining = sorted(
            (
                entry for entry in scored
                if str(entry["id"]) not in selected_ids
            ),
            key=lambda entry: (
                -float(entry.get("base_score") or 0.0),
                str(entry["id"]),
            ),
        )
        for entry in remaining:
            selected_ids.append(str(entry["id"]))
            if len(selected_ids) >= 3:
                break
    selected = [by_id[view_id] for view_id in selected_ids]
    reconstructed = deepcopy(ranking)
    reconstructed["selected_view_ids"] = list(selected_ids)
    reconstructed["fallback_reason"] = fallback_reason
    reconstructed["reconstructed_from_frozen_ranking"] = True
    reconstructed.setdefault(
        "selector",
        "support_contact_plane_visibility_rank_v1",
    )
    return selected, reconstructed


def _minimum_angular_diversity(
    candidate: dict[str, Any],
    selected: list[dict[str, Any]],
) -> float:
    if not selected:
        return 0.0
    azimuth = float(candidate.get("azimuth_degrees") or 0.0)
    elevation = float(candidate.get("elevation_degrees") or 0.0)
    distances: list[float] = []
    for other in selected:
        other_azimuth = float(other.get("azimuth_degrees") or 0.0)
        other_elevation = float(other.get("elevation_degrees") or 0.0)
        azimuth_delta = (
            abs((azimuth - other_azimuth + 180.0) % 360.0 - 180.0) / 180.0
        )
        elevation_delta = min(
            1.0,
            abs(elevation - other_elevation) / 90.0,
        )
        distances.append(
            min(1.0, 0.8 * azimuth_delta + 0.2 * elevation_delta)
        )
    return min(distances)


def _source_contract(
    source: dict[str, Any],
    selected_ids: list[str],
) -> dict[str, Any]:
    return {
        "source_comparison_sha256": _file_sha256(source["comparison_path"]),
        "camera_evidence_manifest_sha256": _file_sha256(
            source["camera_manifest_path"]
        ),
        "pose_candidates_sha256": _file_sha256(source["pose_candidates_path"]),
        "focus_overlay_spec_sha256": _file_sha256(source["overlay_spec_path"]),
        "blend_file_sha256": _file_sha256(source["blend_file"]),
        "selected_top3_view_ids": list(selected_ids),
    }


def _output_ready(path: Path, source_contract: dict[str, Any]) -> bool:
    if not path.is_file():
        return False
    try:
        payload = _read_json(path)
        if (
            payload.get("schema_version") != OUTPUT_SCHEMA
            or payload.get("source_contract") != source_contract
            or tuple((payload.get("arms") or {}).keys()) != ARMS
        ):
            return False
        for arm in ARMS:
            for item in (payload["arms"][arm].get("items") or []):
                item_path = Path(str(item.get("path") or ""))
                if (
                    not item_path.is_file()
                    or _file_sha256(item_path) != item.get("sha256")
                ):
                    return False
        return True
    except Exception:
        return False


def _normalize_item(item: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(item)
    path = Path(str(result.get("path") or "")).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    result["path"] = str(path)
    result["sha256"] = _file_sha256(path)
    return result


def _safe_name(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in value
    ).strip("._") or "event"


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
