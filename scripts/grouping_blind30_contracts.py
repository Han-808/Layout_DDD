#!/usr/bin/env python3
"""Contracts, paths, fingerprints, and safe IO for blind grouping."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

import yaml


EXPERIMENT_SCHEMA_VERSION = "grouping_blind30_experiment_v1"
DATASET_SCHEMA_VERSION = "grouping_blind30_dataset_v1"
RESULT_SCHEMA_VERSION = "grouping_blind30_result_v1"
BLIND_KEY_SCHEMA_VERSION = "grouping_blind30_method_key_v1"
REVIEW_DATA_SCHEMA_VERSION = "grouping_blind30_review_data_v1"
BLIND_LABELS = ("A", "B", "C")


@dataclass(frozen=True)
class ExperimentPaths:
    repo_root: Path
    output_root: Path

    @property
    def dataset_manifest(self) -> Path:
        return self.output_root / "dataset_manifest.json"

    @property
    def experiment_manifest(self) -> Path:
        return self.output_root / "experiment_manifest.json"

    @property
    def method_key(self) -> Path:
        return self.output_root / "private" / "method_key.json"

    @property
    def cases_root(self) -> Path:
        return self.output_root / "cases"

    @property
    def review_root(self) -> Path:
        return self.output_root / "blind_review"

    @property
    def run_summary(self) -> Path:
        return self.output_root / "run_summary.json"

    def case_root(self, case_id: str) -> Path:
        return self.cases_root / case_id


def load_experiment_config(
    config_path: Path,
    *,
    repo_root: Path,
    output_override: Path | None = None,
) -> tuple[dict[str, Any], ExperimentPaths]:
    path = config_path.expanduser().resolve()
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("grouping experiment config must be a YAML object")
    config = deepcopy(value)
    if config.get("schema_version") != EXPERIMENT_SCHEMA_VERSION:
        raise ValueError(
            "grouping experiment config schema_version must be "
            f"{EXPERIMENT_SCHEMA_VERSION!r}"
        )
    experiment_id = required_text(
        config.get("experiment_id"),
        "experiment_id",
    )
    sample = required_object(config.get("sample"), "sample")
    if int(sample.get("size", 0)) != 30:
        raise ValueError("blind grouping experiment sample.size must be 30")
    backends = config.get("backends")
    if not isinstance(backends, list) or set(backends) != {
        "topology",
        "anchor",
        "vlm",
    }:
        raise ValueError(
            "blind grouping experiment backends must contain exactly "
            "topology, anchor, and vlm"
        )
    if len(backends) != 3 or len(set(backends)) != 3:
        raise ValueError("blind grouping experiment backends must be unique")
    seed = config.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("grouping experiment seed must be an integer")
    source_dir = repo_path(
        repo_root,
        required_text(sample.get("source_dir"), "sample.source_dir"),
    )
    if not source_dir.is_dir():
        raise FileNotFoundError(
            f"grouping source scene directory does not exist: {source_dir}"
        )
    config["_resolved_source_dir"] = str(source_dir)
    config["_config_path"] = str(path)
    configured_output = repo_path(
        repo_root,
        required_text(config.get("output_root"), "output_root"),
    )
    output_root = (
        output_override.expanduser().resolve()
        if output_override is not None
        else configured_output
    )
    config["_resolved_output_root"] = str(output_root)
    config["_experiment_id"] = experiment_id
    return config, ExperimentPaths(
        repo_root=repo_root.resolve(),
        output_root=output_root,
    )


def grouping_input_fingerprint(
    *,
    input_manifest: dict[str, Any],
    evidence_paths: list[Path],
    backend: str,
    backend_config: dict[str, Any],
    model_config: dict[str, Any] | None,
) -> str:
    return json_sha256(
        {
            "input_fingerprint": input_manifest["input_fingerprint"],
            "evidence": [
                {
                    "name": path.name,
                    "sha256": file_sha256(path),
                }
                for path in evidence_paths
            ],
            "backend": backend,
            "backend_config": backend_config,
            "model_config": model_config if backend == "vlm" else None,
        }
    )


def load_backend_config(
    config: dict[str, Any],
    *,
    backend: str,
    repo_root: Path,
) -> dict[str, Any]:
    files = required_object(
        config.get("backend_config_files"),
        "backend_config_files",
    )
    path = repo_path(
        repo_root,
        required_text(
            files.get(backend),
            f"backend_config_files.{backend}",
        ),
    )
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"grouping config must be an object: {path}")
    return value


def evidence_packet(
    *,
    input_manifest: dict[str, Any],
    render_manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    views = {
        str(item.get("name")): Path(str(item.get("path"))).resolve()
        for item in render_manifest.get("views", [])
        if isinstance(item, dict)
    }
    required = ("perspective", "top")
    if any(name not in views or not views[name].is_file() for name in required):
        raise FileNotFoundError(
            "render manifest must contain perspective and top images"
        )
    identity_path = Path(input_manifest["identity_map_path"]).resolve()
    aliases = input_manifest["object_aliases"]
    legend = {
        alias: object_id for object_id, alias in aliases.items()
    }
    return [
        {
            "path": str(views["perspective"]),
            "role": "global_perspective_rgb",
            "representation": "rgb",
            "view_id": "global_perspective",
            "object_ids": list(aliases),
            "camera_scope": "global",
        },
        {
            "path": str(views["top"]),
            "role": "global_top_rgb",
            "representation": "rgb",
            "view_id": "global_top",
            "object_ids": list(aliases),
            "camera_scope": "global",
        },
        {
            "path": str(identity_path),
            "role": "global_identity_overlay",
            "representation": "identity_map",
            "view_id": "global_identity",
            "object_ids": list(aliases),
            "identity_overlay": True,
            "identity_legend": legend,
            "camera_scope": "global",
        },
    ]


def blind_label_for_backend(
    method_key: dict[str, Any],
    *,
    case_id: str,
    backend: str,
) -> str:
    case_mapping = method_key.get("cases", {}).get(case_id)
    if not isinstance(case_mapping, dict):
        raise KeyError(f"method key does not contain {case_id}")
    for label, value in case_mapping.items():
        if value == backend:
            return str(label)
    raise KeyError(
        f"method key does not map {backend!r} for {case_id}"
    )


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return value


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        indent=2,
        ensure_ascii=False,
        sort_keys=True,
    )
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(payload)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def sanitized_error(exc: BaseException) -> dict[str, str]:
    text = str(exc)
    lowered = text.lower()
    for token in ("authorization:", "bearer ", "x-api-key"):
        if token in lowered:
            text = "redacted because the exception contained credential-like headers"
            break
    return {
        "error_type": type(exc).__name__,
        "error": text[:2_000],
    }


def repo_path(repo_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (
        path.resolve()
        if path.is_absolute()
        else (repo_root / path).resolve()
    )


def required_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def required_text(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} must be non-empty")
    return text
