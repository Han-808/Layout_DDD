from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False)
        f.write("\n")
    return target


def load_yaml(path: str | Path, default: Any | None = None) -> Any:
    target = Path(path)
    if not target.exists():
        return default
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency declared in pyproject
        raise RuntimeError("PyYAML is required to load YAML configuration files.") from exc
    with target.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    return default if loaded is None else loaded


def load_json_schema(path: str | Path) -> dict:
    schema = read_json(path)
    if not isinstance(schema, dict):
        raise ValueError(f"Schema at {path} must be a JSON object.")
    return schema
