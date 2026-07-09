from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from benchmark.evaluator import evaluate_oor
from benchmark.utils.io import read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate lightweight Object-Object Relationship constraints.")
    parser.add_argument("--scene", required=True, help="Scene JSON containing objects/assets and optional OOR relations.")
    parser.add_argument("--relations", default=None, help="Optional JSON relation spec file. Can be a list or an object with oor_relations/relations.")
    parser.add_argument("--config", default=None, help="Optional JSON config override.")
    parser.add_argument("--out", required=True, help="Output report JSON path.")
    args = parser.parse_args()

    scene = read_json(_path_arg(args.scene))
    relation_specs = _relation_specs_from_file(_path_arg(args.relations)) if args.relations else None
    config = read_json(_path_arg(args.config)) if args.config else None
    if config is not None and not isinstance(config, dict):
        parser.error("--config must point to a JSON object.")
    report = evaluate_oor(scene, relation_specs=relation_specs, config=config)
    out_path = write_json(_path_arg(args.out), report)
    print(f"overall_score: {report['overall_score']}")
    print(f"num_checks_called: {report['num_checks_called']}")
    print(f"num_passed: {report['num_passed']}")
    print(f"num_failed: {report['num_failed']}")
    print(f"report: {out_path}")


def _relation_specs_from_file(path: Path) -> list[dict]:
    loaded = read_json(path)
    if isinstance(loaded, list):
        return [item for item in loaded if isinstance(item, dict)]
    if isinstance(loaded, dict):
        for key in ["oor_relations", "relations"]:
            value = loaded.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    raise ValueError(f"Relation spec file must be a JSON list or contain oor_relations/relations: {path}")


def _path_arg(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path.cwd() / path


if __name__ == "__main__":
    main()
