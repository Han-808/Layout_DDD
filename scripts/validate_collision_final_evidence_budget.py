"""Bounded, local-only old/new Collision acquisition replay; never calls a Judge.

Pass 1-6 existing camera_evidence_manifest.json paths. Both arms render into a
new output root. Source blends/requests and official evaluations are read-only.
The direct Blender binary intentionally excludes production gate/concurrency
timing. Results are evidence equivalence checks, not new metric scores.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def without_paths(value):
    if isinstance(value, dict):
        return {
            key: without_paths(item) for key, item in value.items()
            if key != "path" and not key.endswith("_path")
        }
    if isinstance(value, list):
        return [without_paths(item) for item in value]
    return value


def replay(args):
    sys.path.insert(0, str(Path(args.source_root).resolve() / "src"))
    from benchmark.rendering.blender import BlenderRenderer
    from benchmark.visual_judge.render_views import CameraEvidenceProvider
    from benchmark.visual_judge.visual_config import compose_default_p0b_visual_evidence

    original_path = Path(args.manifest[0]).resolve()
    original = read(original_path)
    policy = original["policy"]
    if original["resolved_mode"] != "visibility_ranked" or not policy["collision_contour"]:
        raise ValueError("replay requires recorded visibility_ranked Collision contour evidence")
    if policy["selector_identity"] is not None or policy["collision_geometry_sha256"] is not None:
        raise ValueError("this local replay supports only deterministic, no-mesh runtime packets")
    output = Path(args.out_root).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_blend = Path(policy["source_blend"])
    before = sha256(source_blend)
    if before != policy["source_blend_sha256"]:
        raise ValueError("source blend no longer matches the recorded packet")
    renderer_config = {key: value for key, value in policy["renderer"].items() if key != "class"}
    renderer_config["blender_bin"] = args.blender_bin
    renderer = BlenderRenderer(**renderer_config)
    calls = []

    def timed(name, method):
        def call(**kwargs):
            started = time.perf_counter()
            record = {
                "method": name,
                "preview": kwargs.get("preview", False),
                "view_ids": [pose["id"] for pose in kwargs["camera_views"]],
                "respect_occlusion": kwargs.get("respect_occlusion"),
            }
            try:
                result = method(**kwargs)
                record["status"] = "complete"
                return result
            except Exception as exc:
                record.update(status="error", error_type=type(exc).__name__)
                raise
            finally:
                record["wall_seconds"] = time.perf_counter() - started
                calls.append(record)
                write(output / "render_calls.json", calls)
                print(json.dumps({"arm": args.arm, **record}), flush=True)
        return call

    for method in ["render_camera_views", "render_focus_overlay_views", "render_target_id_masks"]:
        setattr(renderer, method, timed(method, getattr(renderer, method)))
    options = {key: policy[key] for key in [
        "mode", "max_views", "max_steps", "candidate_count", "candidate_policy",
        "active_repair", "collision_overlay", "collision_contour",
        "frozen_view_ids", "highlighted_global_pose_policy", "architecture_contract",
    ]}
    options["metric_modes"] = policy["metric_mode_overrides"]
    if args.arm == "candidate":
        options["collision_final_view_count"] = 1
    provider = CameraEvidenceProvider(
        renderer=renderer, blend_file=source_blend, out_dir=output / "evidence", **options
    )
    started = time.perf_counter()
    items = provider(read(original_path.parent / "evidence_request.json"))
    elapsed = time.perf_counter() - started
    consumed, visual_policy = compose_default_p0b_visual_evidence("collision", items)
    result = {
        "source_root": str(Path(args.source_root).resolve()),
        "original_manifest": str(original_path),
        "request_sha256": sha256(original_path.parent / "evidence_request.json"),
        "source_blend_sha256_before": before,
        "source_blend_sha256_after": sha256(source_blend),
        "provider_wall_seconds": elapsed,
        "manifest_path": provider.last_call_usage["manifest_path"],
        "consumed_items": consumed,
        "visual_policy": visual_policy,
        "render_calls": calls,
        "judge_calls": 0,
    }
    write(output / "result.json", result)


def compare(baseline, candidate):
    from PIL import Image, ImageChops

    a, b = read(baseline / "result.json"), read(candidate / "result.json")
    am, bm = read(a["manifest_path"]), read(b["manifest_path"])
    pixels = []
    for ai, bi in zip(a["consumed_items"], b["consumed_items"]):
        with Image.open(ai["path"]) as ap, Image.open(bi["path"]) as bp:
            same = (ap.mode, ap.size, ap.tobytes()) == (bp.mode, bp.size, bp.tobytes())
            pixels.append({
                "role": ai["role"], "view_id": ai["view_id"], "equal": same,
                "baseline_decoded_sha256": hashlib.sha256(ap.tobytes()).hexdigest(),
                "candidate_decoded_sha256": hashlib.sha256(bp.tobytes()).hexdigest(),
                "difference_bbox": None if same else ImageChops.difference(ap.convert("RGB"), bp.convert("RGB")).getbbox(),
            })
    checks = {
        "source_blend_unchanged": a["source_blend_sha256_before"] == a["source_blend_sha256_after"] == b["source_blend_sha256_before"] == b["source_blend_sha256_after"],
        "same_request": a["request_sha256"] == b["request_sha256"],
        "same_candidates": read(Path(a["manifest_path"]).parent / "pose_candidates.json") == read(Path(b["manifest_path"]).parent / "pose_candidates.json"),
        "same_selection": am["selection"] == bm["selection"],
        "same_preview_visibility": without_paths(am["candidate_visibility"]) == without_paths(bm["candidate_visibility"]),
        "same_first_pose": am["selected_poses"][:1] == bm["selected_poses"],
        "same_backup_selection": am["selected_poses"] == bm["final_evidence_budget"]["selected_poses_before_final_budget"],
        "same_visual_policy": a["visual_policy"] == b["visual_policy"],
        "same_consumer_metadata": without_paths(a["consumed_items"]) == without_paths(b["consumed_items"]),
        "same_two_decoded_images": len(pixels) == 2 and all(item["equal"] for item in pixels),
    }
    return {
        "passed": all(checks.values()), "checks": checks, "pixels": pixels,
        "baseline_seconds": a["provider_wall_seconds"], "candidate_seconds": b["provider_wall_seconds"],
        "relative_reduction": 1 - b["provider_wall_seconds"] / a["provider_wall_seconds"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--blender-bin", required=True)
    parser.add_argument("--baseline-source")
    parser.add_argument("--candidate-source")
    parser.add_argument("--source-root")
    parser.add_argument("--arm", choices=["baseline", "candidate"])
    args = parser.parse_args()
    if args.arm:
        replay(args)
        return
    if not args.baseline_source or not args.candidate_source or not 1 <= len(args.manifest) <= 6:
        parser.error("provide both source roots and 1-6 fixed manifests")
    root = Path(args.out_root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    plan = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifests": args.manifest, "baseline_source": args.baseline_source,
        "candidate_source": args.candidate_source, "blender_bin": args.blender_bin,
        "judge_calls": 0, "production_gate": False, "results": [],
    }
    write(root / "validation.json", plan)
    for index, manifest in enumerate(args.manifest):
        case = root / f"case_{index + 1:02d}"
        case.mkdir()
        # Alternate order to reduce systematic warm-cache ordering bias.
        arms = ["baseline", "candidate"] if index % 2 == 0 else ["candidate", "baseline"]
        for arm in arms:
            source = args.baseline_source if arm == "baseline" else args.candidate_source
            subprocess.run([
                sys.executable, "-B", str(Path(__file__).resolve()), "--arm", arm,
                "--source-root", source, "--manifest", manifest,
                "--out-root", str(case / arm), "--blender-bin", args.blender_bin,
            ], check=True)
        result = compare(case / "baseline", case / "candidate")
        write(case / "comparison.json", result)
        plan["results"].append({"case": case.name, **result})
        write(root / "validation.json", plan)
        print(json.dumps({"case": case.name, **result}), flush=True)
        if not result["passed"]:
            raise RuntimeError("evidence comparison failed; stop before rendering more cases")
    plan["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    plan["passed"] = True
    write(root / "validation.json", plan)


if __name__ == "__main__":
    main()
