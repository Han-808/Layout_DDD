from __future__ import annotations

LEGEND_INPUT_CHAIN = True
CURRENT_INPUT_CHAIN = "natural_language"

import argparse
import fnmatch
import json
import sys
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

HF_DATASET = "hssd/hssd-hab"
HF_REVISION = "main"
DEFAULT_METADATA_ALLOW_PATTERNS = [
    "*.scene_dataset_config.json",
    "scene_splits.yaml",
    "scenes/**/*.scene_instance.json",
    "scenes-uncluttered/**/*.scene_instance.json",
    "scenes-articulated/**/*.scene_instance.json",
    "metadata/**/*.csv",
    "metadata/**/*.json",
    "metadata/**/*.txt",
    "metadata/**/*.png",
    "metadata/**/*.jpg",
    "metadata/**/*.jpeg",
    "metadata/**/*.webp",
    "semantics/**/*.csv",
    "semantics/**/*.json",
    "semantics/**/*.png",
    "semantics/**/*.jpg",
    "semantics/**/*.jpeg",
    "semantics/**/*.webp",
    "scene_filter_files/**/*.json",
    "stages/**/*.stage_config.json",
    "objects/**/*.object_config.json",
    "urdf/**/*.ao_config.json",
    "urdf/**/*.urdf",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="LEGEND: validate or explicitly download a local HSSD-HAB dataset root.")
    parser.add_argument("--hssd-root", default=str(PROJECT_ROOT / "data" / "external" / "hssd-hab"))
    parser.add_argument("--download", action="store_true", help="Explicitly download hssd/hssd-hab with huggingface_hub.")
    parser.add_argument(
        "--allow-patterns",
        nargs="*",
        default=None,
        help="Optional Hugging Face allow patterns. Defaults to metadata-only patterns, excluding meshes/assets.",
    )
    args = parser.parse_args()

    root = Path(args.hssd_root)
    allow_patterns = args.allow_patterns or DEFAULT_METADATA_ALLOW_PATTERNS
    if args.download:
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            _download_matching_files(root, allow_patterns)
        else:
            snapshot_download(
                repo_id=HF_DATASET,
                repo_type="dataset",
                local_dir=str(root),
                allow_patterns=allow_patterns,
            )

    if not root.exists():
        raise SystemExit(f"HSSD-HAB root does not exist: {root}")
    scene_files = list(root.rglob("*.scene_instance.json"))
    if not scene_files:
        raise SystemExit(f"No *.scene_instance.json files found under: {root}")
    print(f"HSSD-HAB root OK: {root}")
    print(f"scene_instance files: {len(scene_files)}")


def _download_matching_files(root: Path, allow_patterns: list[str] | None) -> None:
    patterns = allow_patterns or ["*.scene_instance.json"]
    prefixes = _prefixes_from_patterns(patterns)
    matched_paths: list[str] = []
    for prefix in prefixes:
        matched_paths.extend(
            item["path"]
            for item in _list_hf_tree(prefix)
            if item.get("type") == "file" and _matches_any(item.get("path", ""), patterns)
        )
    unique_paths = sorted(set(matched_paths))
    if not unique_paths:
        raise SystemExit(f"No files matched allow patterns: {patterns}")
    for relative_path in unique_paths:
        target = root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/{HF_REVISION}/{quote(relative_path)}"
        with urlopen(url, timeout=120) as response:
            target.write_bytes(response.read())
    print(f"Downloaded {len(unique_paths)} files with urllib fallback.")


def _list_hf_tree(prefix: str) -> list[dict]:
    quoted_prefix = quote(prefix.strip("/"))
    suffix = f"/{quoted_prefix}" if quoted_prefix else ""
    url = f"https://huggingface.co/api/datasets/{HF_DATASET}/tree/{HF_REVISION}{suffix}?recursive=true"
    with urlopen(url, timeout=120) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, list):
        raise SystemExit(f"Unexpected Hugging Face tree response for {prefix!r}.")
    return payload


def _prefixes_from_patterns(patterns: list[str]) -> list[str]:
    prefixes = []
    for pattern in patterns:
        parts = pattern.replace("\\", "/").split("/")
        prefix_parts = []
        for part in parts:
            if any(char in part for char in "*?["):
                break
            prefix_parts.append(part)
        prefix = "/".join(prefix_parts)
        prefixes.append(prefix)
    return sorted(set(prefixes or [""]))


def _matches_any(path: str, patterns: list[str]) -> bool:
    normalized = path.replace("\\", "/")
    for pattern in patterns:
        normalized_pattern = pattern.replace("\\", "/")
        if fnmatch.fnmatch(normalized, normalized_pattern):
            return True
        if "/**/" in normalized_pattern and fnmatch.fnmatch(normalized, normalized_pattern.replace("/**/", "/")):
            return True
    return False


if __name__ == "__main__":
    main()
