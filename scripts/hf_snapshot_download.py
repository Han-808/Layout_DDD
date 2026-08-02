#!/usr/bin/env python3
"""Download one registry entry from the Hugging Face Hub at a pinned revision.

Uses the ``snapshot_download`` Python API rather than the command line. The CLI
was renamed from ``huggingface-cli`` to ``hf`` in huggingface_hub 1.0, so
invoking it by module path breaks depending on which version an environment
happens to carry. The API signature has been stable since 0.17.

Usage:
    python scripts/hf_snapshot_download.py --kind models --key PointLLM-R-7B
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

DEFAULT_REGISTRY = (
    Path(__file__).resolve().parents[1] / "configs" / "models" / "pointllm_mnet_registry.json"
)
DEFAULT_MODELS_ROOT = Path("/mnt/group/cmh/models")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--kind", choices=("models", "datasets"), required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--models-root", type=Path, default=DEFAULT_MODELS_ROOT)
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()

    import huggingface_hub
    from huggingface_hub import snapshot_download

    print(f"huggingface_hub {huggingface_hub.__version__}")

    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    section = registry.get(args.kind, {})
    if args.key not in section:
        print(f"unknown {args.kind} key {args.key!r}", file=sys.stderr)
        return 2
    spec = section[args.key]

    declared = Path(spec["local_dir"])
    try:
        relative = declared.relative_to(DEFAULT_MODELS_ROOT)
    except ValueError:
        relative = Path(declared.name)
    local_dir = args.models_root / relative

    kwargs = {
        "repo_id": spec["hf_repo"],
        "revision": spec["revision"],
        "local_dir": str(local_dir),
        "max_workers": args.max_workers,
    }
    if args.kind == "datasets":
        kwargs["repo_type"] = "dataset"
    if spec.get("allow_patterns"):
        kwargs["allow_patterns"] = list(spec["allow_patterns"])

    # Before 0.23 a local_dir download symlinked into the shared cache, which
    # would silently double the 27 GB footprint. The parameter is deprecated in
    # later releases and removed in 1.x, so only pass it when it exists.
    if "local_dir_use_symlinks" in inspect.signature(snapshot_download).parameters:
        kwargs["local_dir_use_symlinks"] = False

    print(f"repo     : {spec['hf_repo']}")
    print(f"revision : {spec['revision']}")
    print(f"local_dir: {local_dir}")

    resolved = snapshot_download(**kwargs)
    print(f"downloaded to {resolved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
