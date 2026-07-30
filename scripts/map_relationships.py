"""Build a relationship-intent skeleton from an object plan.

Summary:
    Maps the relation intents declared in an object plan into the canonical
    ``relationship_intent.json`` structure used by downstream alignment (this is
    non-scoring alignment infrastructure, not a metric).

Input:
    - ``--scene-request`` and ``--object-plan`` JSON.
    - ``--mode``: ``passthrough`` (deterministic) or ``vlm``.

Output:
    - ``--out``: relationship_intent JSON; prints the written path.

Function:
    Thin CLI wrapper over ``map_relationships(...)``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from benchmark.relationship_mapper import map_relationships
from benchmark.utils.io import read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Map object_plan relation intents into relationship_intent.json skeleton output.")
    parser.add_argument("--scene-request", required=True)
    parser.add_argument("--object-plan", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", default="passthrough", choices=["passthrough", "vlm"])
    args = parser.parse_args()

    result = map_relationships(
        scene_request=read_json(_path_arg(args.scene_request)),
        object_plan=read_json(_path_arg(args.object_plan)),
        mode=args.mode,
    )
    out_path = write_json(_path_arg(args.out), result)
    print(f"relationship_intent: {out_path}")


def _path_arg(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path.cwd() / path


if __name__ == "__main__":
    main()
