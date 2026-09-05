#!/usr/bin/env python3
"""Stage-1 verification for PointLLM checkpoints downloaded onto MNET.

This checks the checkpoint on disk only. It never starts a server and never
runs inference, so a pass here means "the bytes are complete and pinned", not
"the model works". Stage 2 (environment) and stage 3 (real point-cloud
inference) are separate and must be checked separately.

Usage:
    python scripts/verify_pointllm_checkpoint.py --model all
    python scripts/verify_pointllm_checkpoint.py --model PointLLM-R-7B --sha256 full
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

DEFAULT_REGISTRY = (
    Path(__file__).resolve().parents[1] / "configs" / "models" / "pointllm_mnet_registry.json"
)

# huggingface_hub writes resume metadata under <local_dir>/.cache/huggingface.
# It is bookkeeping, not part of the checkpoint, and must not affect the count.
IGNORED_DIR_NAMES = {".cache", ".git"}
INCOMPLETE_SUFFIXES = (".incomplete", ".part", ".tmp", ".lock")

CHUNK = 16 * 1024 * 1024


class Failure(Exception):
    pass


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(CHUNK)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def iter_checkpoint_files(root: Path):
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if IGNORED_DIR_NAMES.intersection(relative.parts):
            continue
        yield relative, path


def verify_model(name: str, spec: dict, models_root: Path | None, sha_mode: str) -> dict:
    problems: list[str] = []
    local_dir = Path(spec["local_dir"])
    if models_root is not None:
        local_dir = models_root / local_dir.name

    report = {
        "model": name,
        "hf_repo": spec["hf_repo"],
        "revision": spec["revision"],
        "local_dir": str(local_dir),
        "sha256_mode": sha_mode,
        "status": "fail",
        "problems": problems,
    }

    if not local_dir.is_dir():
        problems.append(f"missing checkpoint directory: {local_dir}")
        return report

    expected_files = spec["files"]
    present = dict(iter_checkpoint_files(local_dir))
    present_names = {str(rel) for rel in present}

    report["file_count"] = f"{len(present_names)}/{spec['expected_file_count']}"

    missing = sorted(set(expected_files) - present_names)
    unexpected = sorted(present_names - set(expected_files))
    for item in missing:
        problems.append(f"missing file: {item}")
    for item in unexpected:
        if item.endswith(INCOMPLETE_SUFFIXES):
            problems.append(f"partial download artifact still present: {item}")
        else:
            problems.append(f"unexpected extra file: {item}")

    total_bytes = 0
    checked_digests = 0
    for filename, meta in sorted(expected_files.items()):
        relative = Path(filename)
        if str(relative) not in present_names:
            continue
        path = present[relative]
        actual_size = path.stat().st_size
        total_bytes += actual_size
        if actual_size != meta["size"]:
            problems.append(
                f"size mismatch: {filename} is {actual_size} bytes, expected {meta['size']}"
            )
            continue
        expected_sha = meta.get("sha256")
        want_sha = sha_mode == "full" or (sha_mode == "lfs" and expected_sha is not None)
        if want_sha and expected_sha is not None:
            actual_sha = sha256_of(path)
            checked_digests += 1
            if actual_sha != expected_sha:
                problems.append(
                    f"sha256 mismatch: {filename} is {actual_sha}, expected {expected_sha}"
                )

    report["total_bytes"] = total_bytes
    report["sha256_files_checked"] = checked_digests
    expected_total = sum(meta["size"] for meta in expected_files.values())
    if total_bytes != expected_total and not missing:
        problems.append(
            f"total size mismatch: {total_bytes} bytes on disk, expected {expected_total}"
        )
    # Catches drift between the registry's documented total and its file map.
    if spec["total_bytes"] != expected_total:
        problems.append(
            f"registry is self-inconsistent: total_bytes={spec['total_bytes']} "
            f"but the file map sums to {expected_total}"
        )

    def read_json(path: Path) -> dict | None:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            problems.append(f"unreadable JSON: {path.name} ({error})")
            return None

    config_path = local_dir / "config.json"
    if config_path.is_file():
        config = read_json(config_path) or {}
        report["model_type"] = config.get("model_type")
        report["architectures"] = config.get("architectures")
        if config.get("model_type") != spec["model_type"]:
            problems.append(
                f"config model_type is {config.get('model_type')!r}, expected {spec['model_type']!r}"
            )
        if config.get("architectures") != spec["architectures"]:
            problems.append(
                f"config architectures is {config.get('architectures')!r}, "
                f"expected {spec['architectures']!r}"
            )
        if config.get("vocab_size") != spec["vocab_size"]:
            problems.append(
                f"config vocab_size is {config.get('vocab_size')!r}, expected {spec['vocab_size']}"
            )

    index_path = local_dir / "pytorch_model.bin.index.json"
    if index_path.is_file():
        index = read_json(index_path) or {}
        weight_map = index.get("weight_map", {})
        index_total = index.get("metadata", {}).get("total_size")
        report["index_tensor_count"] = len(weight_map)
        report["index_total_size"] = index_total
        if len(weight_map) != spec["index_tensor_count"]:
            problems.append(
                f"index lists {len(weight_map)} tensors, expected {spec['index_tensor_count']}"
            )
        if index_total != spec["index_total_size"]:
            problems.append(
                f"index total_size is {index_total}, expected {spec['index_total_size']}"
            )
        # The point encoder and projector are fused into the shards; a
        # checkpoint without them would load as a plain Llama and silently
        # ignore the point cloud.
        for prefix in ("model.point_backbone.", "model.point_proj."):
            count = sum(1 for key in weight_map if key.startswith(prefix))
            report[f"tensors_{prefix.strip('.').replace('.', '_')}"] = count
            if count == 0:
                problems.append(f"no {prefix}* tensors in the weight index")
        referenced = set(weight_map.values())
        for shard in sorted(referenced):
            if not (local_dir / shard).is_file():
                problems.append(f"weight index references a missing shard: {shard}")

    report["status"] = "fail" if problems else "pass"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help="model key from the registry, or 'all' (default: all)",
    )
    parser.add_argument(
        "--models-root",
        type=Path,
        default=None,
        help="override the parent directory of every checkpoint (for local dry runs)",
    )
    parser.add_argument(
        "--sha256",
        choices=("skip", "lfs", "full"),
        default="lfs",
        help=(
            "'lfs' hashes only the large weight files with a published digest (default), "
            "'full' is identical here but explicit, 'skip' checks sizes only"
        ),
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    all_models = registry["models"]

    selected = args.model or ["all"]
    if "all" in selected:
        keys = list(all_models)
    else:
        keys = selected
        unknown = sorted(set(keys) - set(all_models))
        if unknown:
            print(f"unknown model keys: {unknown}", file=sys.stderr)
            return 2

    reports = [verify_model(key, all_models[key], args.models_root, args.sha256) for key in keys]

    for report in reports:
        print(f"==== {report['model']} ====")
        print(f"  hf_repo         : {report['hf_repo']}")
        print(f"  revision        : {report['revision']}")
        print(f"  local_dir       : {report['local_dir']}")
        print(f"  files           : {report.get('file_count', 'n/a')}")
        if "total_bytes" in report:
            print(f"  bytes on disk   : {report['total_bytes']} ({report['total_bytes'] / 1e9:.2f} GB)")
        print(f"  sha256 verified : {report.get('sha256_files_checked', 0)} file(s) [{report['sha256_mode']}]")
        print(f"  index tensors   : {report.get('index_tensor_count', 'n/a')}")
        print(
            "  point encoder   : "
            f"backbone={report.get('tensors_model_point_backbone', 'n/a')} "
            f"proj={report.get('tensors_model_point_proj', 'n/a')}"
        )
        for problem in report["problems"]:
            print(f"  PROBLEM: {problem}")
        print(f"  status          : {report['status'].upper()}")

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(reports, indent=2), encoding="utf-8")
        print(f"wrote {args.json_out}")

    failed = [report["model"] for report in reports if report["status"] != "pass"]
    if failed:
        print(f"\nSTAGE 1 FAILED for: {', '.join(failed)}", file=sys.stderr)
        return 1
    print("\nSTAGE 1 PASSED: checkpoints are complete and match pinned revisions.")
    print("This says nothing about the runtime environment or inference. Run stages 2 and 3.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
