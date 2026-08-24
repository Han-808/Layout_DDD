#!/usr/bin/env python3
"""Build a non-destructive, model-grouped view of multi-room generations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "scene10_all_models_v1"
API2_ROOT = (
    REPO_ROOT
    / "Support/artifacts/outputs/e2e_multi_room/"
    "api2_gpt56sol_kimi_k3_glm53_scene10_r1"
)
API3_ROOT = (
    REPO_ROOT
    / "Support/artifacts/outputs/e2e_multi_room/"
    "api3_opus5_sonnet5_fable5_provider_default_scene10_r1"
)
API3_RETRY_R2 = (
    REPO_ROOT
    / "Support/artifacts/outputs/e2e_multi_room/"
    "api3_opus5_sonnet5_fable5_failed12_retry_r2"
)
API3_SONNET_R3 = (
    REPO_ROOT
    / "Support/artifacts/outputs/e2e_multi_room/"
    "api3_sonnet5_unresolved4_retry_r3"
)
FLOOR_PLAN_ROOT = REPO_ROOT / "output/multi_room_generation_handoff_v1"

MODEL_SOURCES = {
    "gpt-5.6-sol": API2_ROOT / "gpt-5.6-sol",
    "kimi-k3": API2_ROOT / "kimi-k3",
    "glm-5.3": API2_ROOT / "glm-5.3",
    "claude-opus-5-aihub": API3_ROOT / "claude-opus-5-aihub",
    "claude-sonnet-5-aihub": API3_ROOT / "claude-sonnet-5-aihub",
    "claude-fable-5-aihub": API3_ROOT / "claude-fable-5-aihub",
}
API3_MODELS = {
    "claude-opus-5-aihub",
    "claude-sonnet-5-aihub",
    "claude-fable-5-aihub",
}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_path(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _replace_symlink(path: Path, target: Path) -> None:
    if path.is_symlink():
        path.unlink()
    elif path.exists():
        raise FileExistsError(f"refusing to replace non-symlink: {path}")
    path.symlink_to(os.path.relpath(target, start=path.parent))


def _expected_room_keys(layout_id: str) -> list[str]:
    floor_plan = _json(FLOOR_PLAN_ROOT / layout_id / "floor_plan.json")
    count = int(floor_plan["room_count"])
    return [f"room_{index:03d}" for index in range(count)]


def _retry_candidates(model: str, layout_id: str) -> list[tuple[str, Path]]:
    if model not in API3_MODELS:
        return []
    candidates: list[tuple[str, Path]] = []
    for chance in range(1, 4):
        path = (
            API3_RETRY_R2
            / f"chance_{chance:02d}/outputs"
            / model
            / layout_id
        )
        if (path / "summary.json").is_file():
            candidates.append((f"api3_failed12_r2_chance_{chance:02d}", path))
    return candidates


def _local_retry_candidates(
    model_root: Path, model: str, layout_id: str
) -> list[tuple[str, Path]]:
    retries = model_root / "retries"
    if not retries.is_dir():
        return []
    result: list[tuple[str, Path]] = []
    for retry_root in sorted(retries.iterdir(), key=lambda item: item.name):
        if not retry_root.is_dir() or retry_root.is_symlink():
            continue
        patterns = (
            retry_root / layout_id,
            retry_root / "outputs" / model / layout_id,
        )
        for candidate in patterns:
            if (candidate / "summary.json").is_file():
                result.append((f"local_retry:{retry_root.name}", candidate))
        for chance in sorted(retry_root.glob("chance_*")):
            candidate = chance / "outputs" / model / layout_id
            if (candidate / "summary.json").is_file():
                result.append(
                    (f"local_retry:{retry_root.name}/{chance.name}", candidate)
                )
    return result


def _select_layout(
    model: str, layout_id: str, base: Path, model_root: Path
) -> tuple[str, Path] | None:
    base_layout = base / layout_id
    candidates = [
        ("base", base_layout),
        *_retry_candidates(model, layout_id),
        *_local_retry_candidates(model_root, model, layout_id),
    ]
    present = [item for item in candidates if (item[1] / "summary.json").is_file()]
    for label, path in present:
        summary = _json(path / "summary.json")
        if summary.get("assembly_status") == "complete":
            return label, path
    return present[0] if present else None


def _room_rows(summary: dict[str, Any], expected: list[str]) -> list[dict[str, Any]]:
    rows = summary.get("results")
    by_key = {
        str(row.get("room_key")): row
        for row in rows
        if isinstance(rows, list) and isinstance(row, dict)
    } if isinstance(rows, list) else {}
    result: list[dict[str, Any]] = []
    for room_key in expected:
        row = by_key.get(room_key)
        if row is None:
            result.append(
                {"room_key": room_key, "status": "missing", "eligible": False}
            )
            continue
        result.append(
            {
                "room_key": room_key,
                "status": row.get("status"),
                "eligible": row.get("eligible_for_room_projection") is True,
                "error_type": row.get("error_type"),
            }
        )
    return result


def _link_history(model_root: Path, model: str) -> None:
    retries = model_root / "retries"
    retries.mkdir(exist_ok=True)
    if model in API3_MODELS and API3_RETRY_R2.is_dir():
        _replace_symlink(retries / "api3_failed12_retry_r2", API3_RETRY_R2)
    if model == "claude-sonnet-5-aihub" and API3_SONNET_R3.is_dir():
        _replace_symlink(retries / "api3_sonnet_unresolved4_retry_r3", API3_SONNET_R3)


def build(output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    model_manifests: list[dict[str, Any]] = []
    for model, base in MODEL_SOURCES.items():
        if not base.is_dir():
            raise FileNotFoundError(f"model base output is absent: {base}")
        model_root = output_root / model
        model_root.mkdir(exist_ok=True)
        selected_root = model_root / "selected"
        selected_root.mkdir(exist_ok=True)
        _replace_symlink(model_root / "base", base)
        _link_history(model_root, model)

        selected_layouts: list[dict[str, Any]] = []
        unresolved_rooms: list[dict[str, str]] = []
        successful_rooms = 0
        failed_rooms = 0
        missing_rooms = 0
        complete_layouts = 0
        terminal_layouts = 0
        for index in range(1, 11):
            layout_id = f"layout_{index:02d}"
            expected = _expected_room_keys(layout_id)
            selected = _select_layout(model, layout_id, base, model_root)
            link = selected_root / layout_id
            if selected is None:
                if link.is_symlink():
                    link.unlink()
                missing_rooms += len(expected)
                unresolved_rooms.extend(
                    {
                        "layout_id": layout_id,
                        "room_key": room_key,
                        "status": "missing_layout",
                    }
                    for room_key in expected
                )
                selected_layouts.append(
                    {
                        "layout_id": layout_id,
                        "selection_status": "missing",
                        "expected_rooms": len(expected),
                    }
                )
                continue
            source_label, source = selected
            _replace_symlink(link, source)
            summary_path = source / "summary.json"
            summary = _json(summary_path)
            layout_root = source / layout_id
            evaluation_index_path = layout_root / "room_evaluation_index.json"
            assembly_manifest_path = layout_root / "assembly_manifest.json"
            if not evaluation_index_path.is_file() or not assembly_manifest_path.is_file():
                raise FileNotFoundError(
                    f"selected layout lacks trusted assembly artifacts: {model}/{layout_id}"
                )
            rooms = _room_rows(summary, expected)
            successes = sum(row["eligible"] and row["status"] == "complete" for row in rooms)
            missing = sum(row["status"] == "missing" for row in rooms)
            failures = len(rooms) - successes - missing
            successful_rooms += successes
            failed_rooms += failures
            missing_rooms += missing
            terminal_layouts += 1
            is_complete = summary.get("assembly_status") == "complete" and successes == len(expected)
            complete_layouts += int(is_complete)
            for row in rooms:
                if not (row["eligible"] and row["status"] == "complete"):
                    unresolved_rooms.append(
                        {
                            "layout_id": layout_id,
                            "room_key": row["room_key"],
                            "status": str(row["status"]),
                            "error_type": str(row.get("error_type") or ""),
                        }
                    )
            selected_layouts.append(
                {
                    "layout_id": layout_id,
                    "selection_status": "complete" if is_complete else "incomplete",
                    "source_kind": source_label,
                    "source_path": _repo_path(source),
                    "summary_sha256": _sha256(summary_path),
                    "room_evaluation_index_sha256": _sha256(
                        evaluation_index_path
                    ),
                    "assembly_manifest_sha256": _sha256(
                        assembly_manifest_path
                    ),
                    "expected_rooms": len(expected),
                    "successful_rooms": successes,
                    "failed_rooms": failures,
                    "missing_rooms": missing,
                }
            )

        manifest = {
            "schema_version": "multi_room_model_selection_manifest_v1",
            "model": model,
            "base_source": _repo_path(base),
            "selection_policy": "first_complete_layout_then_base_terminal",
            "future_retry_root": _repo_path(model_root / "retries"),
            "expected_layouts": 10,
            "terminal_layouts": terminal_layouts,
            "complete_layouts": complete_layouts,
            "incomplete_layouts": terminal_layouts - complete_layouts,
            "missing_layouts": 10 - terminal_layouts,
            "expected_rooms": 31,
            "successful_rooms": successful_rooms,
            "failed_rooms": failed_rooms,
            "missing_rooms": missing_rooms,
            "selected_layouts": selected_layouts,
            "unresolved_rooms": unresolved_rooms,
        }
        _write_json(model_root / "selection_manifest.json", manifest)
        model_manifests.append(manifest)

    root_manifest = {
        "schema_version": "multi_room_unified_collection_v1",
        "collection_id": "scene10-all-models-v1",
        "model_count": len(model_manifests),
        "models": [item["model"] for item in model_manifests],
        "expected_rooms": sum(item["expected_rooms"] for item in model_manifests),
        "successful_rooms": sum(item["successful_rooms"] for item in model_manifests),
        "failed_rooms": sum(item["failed_rooms"] for item in model_manifests),
        "missing_rooms": sum(item["missing_rooms"] for item in model_manifests),
        "model_manifests": {
            item["model"]: f"{item['model']}/selection_manifest.json"
            for item in model_manifests
        },
        "model_manifest_sha256": {
            item["model"]: _sha256(
                output_root / item["model"] / "selection_manifest.json"
            )
            for item in model_manifests
        },
    }
    _write_json(output_root / "collection_manifest.json", root_manifest)
    (output_root / "README.md").write_text(
        "# Multi-room Scene10 unified model results\n\n"
        "Each model directory contains:\n\n"
        "- `base`: symlink to the immutable original run;\n"
        "- `selected/layout_XX`: current selected layout view;\n"
        "- `retries/<retry_id>`: all future retry outputs for that model;\n"
        "- `selection_manifest.json`: source paths, hashes, and unresolved rooms.\n\n"
        "Do not overwrite `base` or selected source artifacts. Place every new "
        "retry under the matching model's `retries/` directory, then refresh "
        "the selected view with:\n\n"
        "```bash\n"
        "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python "
        "Support/scripts/build_multi_room_unified_collection.py\n"
        "```\n",
        encoding="utf-8",
    )
    return root_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_root.expanduser().resolve()
    try:
        output.relative_to(REPO_ROOT)
        result = build(output)
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}")
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
