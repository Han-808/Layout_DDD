from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate or explicitly download a local HSSD-HAB dataset root.")
    parser.add_argument("--hssd-root", default=str(PROJECT_ROOT / "data" / "external" / "hssd-hab"))
    parser.add_argument("--download", action="store_true", help="Explicitly download hssd/hssd-hab with huggingface_hub.")
    parser.add_argument("--allow-patterns", nargs="*", default=None, help="Optional Hugging Face allow patterns.")
    args = parser.parse_args()

    root = Path(args.hssd_root)
    if args.download:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise SystemExit("Install huggingface_hub to use --download, or provide an existing --hssd-root.") from exc

        snapshot_download(
            repo_id="hssd/hssd-hab",
            repo_type="dataset",
            local_dir=str(root),
            allow_patterns=args.allow_patterns,
        )

    if not root.exists():
        raise SystemExit(f"HSSD-HAB root does not exist: {root}")
    scene_files = list(root.rglob("*.scene_instance.json"))
    if not scene_files:
        raise SystemExit(f"No *.scene_instance.json files found under: {root}")
    print(f"HSSD-HAB root OK: {root}")
    print(f"scene_instance files: {len(scene_files)}")


if __name__ == "__main__":
    main()
