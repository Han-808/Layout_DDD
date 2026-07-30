#!/usr/bin/env python3
"""Map downloaded P0b experiment pictures to cases and judge events.

The transfer should preserve paths relative to the remote combined-run root.
This script keeps two views of the evidence:

* ``all_picture_inventory.csv`` records every downloaded PNG.
* ``judge_picture_mapping.csv`` records the pictures actually supplied to each
  event judgment, including missing/unresolved packets.

It intentionally uses only the Python standard library so it can be run on the
local Mac without the benchmark environment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import struct
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_REMOTE_RUN_ROOT = "/mnt/group/cmh/Layout_DDD/outputs/p0b_combined_20260717_202533"
CAMERA_ARMS = {
    "global_raw",
    "visibility_raw",
    "visibility_highlight",
    "visibility_highlight_global",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis-root",
        type=Path,
        default=Path("Support/artifacts/result/benchmark_metric_analysis"),
        help="Local JSON/TSV analysis root.",
    )
    parser.add_argument(
        "--pictures-root",
        type=Path,
        default=Path("Support/artifacts/result/benchmark_metric_analysis/pictures"),
        help="Downloaded run-relative image tree.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("Support/artifacts/result/benchmark_metric_analysis/picture_mapping"),
    )
    parser.add_argument("--remote-run-root", default=DEFAULT_REMOTE_RUN_ROOT)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _run_relative(remote_path: str, remote_root: str) -> str:
    normalized_root = remote_root.rstrip("/") + "/"
    if remote_path.startswith(normalized_root):
        return remote_path[len(normalized_root) :]
    return remote_path.lstrip("/")


def _event_context(path: Path) -> tuple[str, str]:
    # .../<case>/<arm>/events/<event>.json
    return path.parents[2].name, path.parents[1].name


def _metadata_by_path(request: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in request.get("local_render_evidence_metadata") or []:
        if isinstance(item, dict) and isinstance(item.get("path"), str):
            result[item["path"]] = item
    return result


def _build_judge_rows(
    analysis_root: Path,
    pictures_root: Path,
    remote_root: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    event_glob = "source_distortion5/ablation/*/*/events/*.json"
    for path in sorted(analysis_root.glob(event_glob)):
        data = _read_json(path)
        case_id, arm_from_path = _event_context(path)
        judgement = data.get("judgement") if isinstance(data.get("judgement"), dict) else {}
        request = judgement.get("request") if isinstance(judgement.get("request"), dict) else {}
        metadata = _metadata_by_path(request)
        remote_images = [
            str(value)
            for value in request.get("render_evidence") or []
            if isinstance(value, str) and value.lower().endswith(".png")
        ]
        base = {
            "case_id": case_id,
            "arm": str(data.get("arm") or arm_from_path),
            "camera_mode": str(data.get("camera_mode") or data.get("mode") or ""),
            "evidence_style": str(data.get("evidence_style") or ""),
            "metric": str(data.get("metric") or ""),
            "event_id": str(data.get("event_id") or ""),
            "object_ids": "|".join(str(value) for value in data.get("object_ids") or []),
            "gt_label": str(data.get("gt_label") or ""),
            "predicted_label": str(data.get("predicted_label") or ""),
            "resolved": int(str(data.get("predicted_label") or "") in {"valid", "invalid"}),
            "event_json": str(path),
        }
        if not remote_images:
            rows.append(
                {
                    **base,
                    "image_index": "",
                    "role": "",
                    "view_id": "",
                    "remote_image_path": "",
                    "run_relative_path": "",
                    "local_image_path": "",
                    "local_exists": 0,
                }
            )
            continue
        for index, remote_path in enumerate(remote_images):
            relative = _run_relative(remote_path, remote_root)
            local_path = pictures_root / relative
            item = metadata.get(remote_path, {})
            rows.append(
                {
                    **base,
                    "image_index": index,
                    "role": str(item.get("role") or _infer_role(remote_path)),
                    "view_id": str(item.get("view_id") or Path(remote_path).stem),
                    "remote_image_path": remote_path,
                    "run_relative_path": relative,
                    "local_image_path": str(local_path),
                    "local_exists": int(local_path.is_file()),
                }
            )
    return rows


def _infer_role(path: str) -> str:
    name = Path(path).name
    if "standardized_top" in name:
        return "global_top"
    if "standardized_perspective" in name:
        return "global_perspective"
    if "/final_overlay/" in path:
        return "metric_highlight_local"
    if "/highlighted_global/" in path:
        return "metric_highlighted_global"
    if "/final_rgb/" in path:
        return "metric_local_rgb"
    return "unclassified"


def _png_dimensions(path: Path) -> tuple[int | None, int | None]:
    try:
        with path.open("rb") as handle:
            header = handle.read(24)
        if header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
            return None, None
        return struct.unpack(">II", header[16:24])
    except OSError:
        return None, None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_context(relative: Path) -> dict[str, str]:
    parts = relative.parts
    result = {
        "experiment_arm": parts[0] if parts else "",
        "case_id": "",
        "camera_arm": "",
        "event_key": "",
        "render_stage": relative.parent.name,
    }
    if len(parts) >= 3 and parts[0] == "source_distortion5":
        if parts[1] in {"source_reports", "ablation"}:
            result["case_id"] = parts[2]
        if parts[1] == "ablation" and len(parts) >= 4 and parts[3] in CAMERA_ARMS:
            result["camera_arm"] = parts[3]
        if "camera_evidence" in parts:
            index = parts.index("camera_evidence")
            if index + 1 < len(parts):
                result["event_key"] = parts[index + 1]
    elif len(parts) >= 3 and parts[0] == "generated_ablation10":
        if parts[1] in {"frozen_cases", "ablation"}:
            result["case_id"] = parts[2]
    return result


def _build_inventory_rows(
    pictures_root: Path,
    references: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(pictures_root.rglob("*.png")):
        relative = path.relative_to(pictures_root)
        width, height = _png_dimensions(path)
        refs = references.get(relative.as_posix(), [])
        contexts = _path_context(relative)
        rows.append(
            {
                **contexts,
                "run_relative_path": relative.as_posix(),
                "local_image_path": str(path),
                "filename": path.name,
                "width": width or "",
                "height": height or "",
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "judge_reference_count": len(refs),
                "judge_references": ";".join(
                    f"{item['case_id']}:{item['arm']}:{item['metric']}:{item['event_id']}:{item['role']}"
                    for item in refs
                ),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def _write_summary(
    path: Path,
    judge_rows: list[dict[str, Any]],
    inventory_rows: list[dict[str, Any]],
) -> None:
    event_keys = {
        (row["case_id"], row["arm"], row["metric"], row["event_id"])
        for row in judge_rows
    }
    expected = [row for row in judge_rows if row["remote_image_path"]]
    missing = [row for row in expected if not row["local_exists"]]
    mode_counts = Counter(row["arm"] for row in expected)
    lines = [
        "# P0b Picture Mapping",
        "",
        f"- Event packets mapped: {len(event_keys)}",
        f"- Judge-image references expected: {len(expected)}",
        f"- Expected judge images missing locally: {len(missing)}",
        f"- Downloaded PNG files inventoried: {len(inventory_rows)}",
        "",
        "## Judge-image references by arm",
        "",
    ]
    for mode, count in sorted(mode_counts.items()):
        lines.append(f"- `{mode}`: {count}")
    lines.extend(
        [
            "",
            "`judge_picture_mapping.csv` is the authoritative event-to-image mapping.",
            "`all_picture_inventory.csv` includes diagnostic and preview renders that may not",
            "have been supplied to the judge. Shared files retain every judge reference.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    judge_rows = _build_judge_rows(
        args.analysis_root,
        args.pictures_root,
        args.remote_run_root,
    )
    references: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in judge_rows:
        if row["run_relative_path"]:
            references[str(row["run_relative_path"])].append(row)
    inventory_rows = _build_inventory_rows(args.pictures_root, references)
    _write_csv(args.out_dir / "judge_picture_mapping.csv", judge_rows)
    _write_csv(args.out_dir / "all_picture_inventory.csv", inventory_rows)
    _write_summary(args.out_dir / "README.md", judge_rows, inventory_rows)
    print(f"mapped event rows: {len(judge_rows)}")
    print(f"downloaded PNG files: {len(inventory_rows)}")
    print(f"output: {args.out_dir}")


if __name__ == "__main__":
    main()
