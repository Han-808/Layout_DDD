#!/usr/bin/env python3
"""Arrange the cal_dataset1 camera audit into a human-review folder tree."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any


METRIC_DIRS = {
    "collision": "01_collision",
    "oob": "02_oob",
    "support": "03_support",
}
SEVERITY_DIRS = {
    "obvious": "01_obvious",
    "subtle": "02_subtle",
}


def main() -> None:
    args = _parse_args()
    source_root = Path(args.source_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    if out_dir.exists():
        raise FileExistsError(
            f"review folder already exists: {out_dir}; choose a new --out-dir"
        )
    manifests = sorted(
        source_root.glob("cases/*/events/*/comparison_manifest.json")
    )
    if not manifests:
        raise FileNotFoundError(
            f"no comparison manifests under {source_root / 'cases'}"
        )
    overview_dir = out_dir / "00_overview"
    overview_dir.mkdir(parents=True)
    _copy_overview(source_root, overview_dir)

    index_rows: list[dict[str, Any]] = []
    for manifest_path in manifests:
        comparison = _read_json(manifest_path)
        metric = str(comparison["metric"])
        severity = str(comparison.get("severity_class") or "unknown")
        event_name = (
            f"{comparison['case_id']}__{_safe_name(str(comparison['event_id']))}"
        )
        event_dir = (
            out_dir
            / METRIC_DIRS.get(metric, f"99_{_safe_name(metric)}")
            / SEVERITY_DIRS.get(severity, f"99_{_safe_name(severity)}")
            / event_name
        )
        event_dir.mkdir(parents=True)
        image_rows = _arrange_event_images(comparison, event_dir)
        selector = comparison["arms"]["metric_local_highlight"].get("selection") or {}
        summary = {
            "schema_version": "cal_dataset1_camera_evidence_review_event_v1",
            "case_id": comparison["case_id"],
            "split": comparison["split"],
            "severity_class": severity,
            "metric": metric,
            "event_id": comparison["event_id"],
            "object_ids": comparison.get("object_ids") or [],
            "semantic_label": comparison.get("semantic_label"),
            "gt_basis": comparison.get("gt_basis"),
            "presentation": comparison.get("presentation"),
            "image_budget_per_arm": comparison.get("image_budget_per_arm"),
            "fixed_global_diagnostic_targets_visible": comparison["arms"][
                "fixed_global_highlight"
            ].get("diagnostic_all_targets_visible_somewhere"),
            "metric_local_diagnostic_targets_visible": comparison["arms"][
                "metric_local_highlight"
            ].get("diagnostic_all_targets_visible_somewhere"),
            "metric_local_selection": selector,
            "human_review": {
                "fixed_global_highlight": None,
                "metric_local_highlight": None,
                "allowed_values": ["sufficient", "insufficient", "unclear"],
                "notes": None,
            },
            "images": image_rows,
            "source_comparison_manifest": str(manifest_path),
        }
        (event_dir / "event_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        shutil.copy2(manifest_path, event_dir / "comparison_manifest.json")
        index_rows.append(
            {
                "metric": metric,
                "severity_class": severity,
                "case_id": comparison["case_id"],
                "event_id": comparison["event_id"],
                "object_ids": ",".join(comparison.get("object_ids") or []),
                "fixed_global_review": "",
                "metric_local_review": "",
                "notes": "",
                "event_folder": str(event_dir.relative_to(out_dir)),
            }
        )

    metric_order = {metric: index for index, metric in enumerate(METRIC_DIRS)}
    severity_order = {severity: index for index, severity in enumerate(SEVERITY_DIRS)}
    index_rows.sort(
        key=lambda row: (
            metric_order.get(str(row["metric"]), 99),
            severity_order.get(str(row["severity_class"]), 99),
            str(row["case_id"]),
            str(row["event_id"]),
        )
    )
    _write_index(out_dir / "review_index.tsv", index_rows)
    _write_readme(out_dir, index_rows)
    print(
        json.dumps(
            {
                "review_folder": str(out_dir),
                "event_count": len(index_rows),
                "metric_counts": {
                    metric: sum(row["metric"] == metric for row in index_rows)
                    for metric in METRIC_DIRS
                },
                "severity_counts": {
                    severity: sum(
                        row["severity_class"] == severity for row in index_rows
                    )
                    for severity in SEVERITY_DIRS
                },
            },
            indent=2,
        )
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        default="Support/artifacts/outputs/cal_dataset1_camera_evidence",
    )
    parser.add_argument(
        "--out-dir",
        default="Support/artifacts/outputs/cal_dataset1_camera_evidence/review_grouped",
    )
    return parser.parse_args()


def _copy_overview(source_root: Path, overview_dir: Path) -> None:
    html_candidates = {
        "00_all_events.html": source_root / "index.html",
        "01_collision.html": source_root / "collision.html",
        "02_oob.html": source_root / "oob.html",
        "03_support.html": source_root / "support.html",
    }
    for name, source in html_candidates.items():
        if source.is_file():
            # The original pages live at source_root and reference cases/... .
            # The grouped copies are two levels deeper.
            content = source.read_text(encoding="utf-8").replace(
                'src="cases/',
                'src="../../cases/',
            )
            (overview_dir / name).write_text(content, encoding="utf-8")
    candidates = {
        "review.tsv": source_root / "review.tsv",
        "collision_contact_sheet.png": source_root / "review" / "collision.png",
        "oob_contact_sheet.png": source_root / "review" / "oob.png",
        "support_contact_sheet.png": source_root / "review" / "support.png",
    }
    for name, source in candidates.items():
        if source.is_file():
            shutil.copy2(source, overview_dir / name)


def _arrange_event_images(
    comparison: dict[str, Any],
    event_dir: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    arms = (
        ("FG", "fixed_global_highlight"),
        ("ML", "metric_local_highlight"),
    )
    image_number = 0
    for arm_code, arm_name in arms:
        items = comparison["arms"][arm_name].get("items") or []
        for item in items:
            image_number += 1
            source = Path(str(item["path"])).expanduser().resolve()
            presentation = (
                "raw"
                if str(item.get("role")) in {"metric_local_rgb", "collision_rgb"}
                else "highlight"
            )
            view_id = _safe_name(str(item.get("view_id") or "view"))
            destination = event_dir / (
                f"{image_number:02d}_{arm_code}_{view_id}_{presentation}{source.suffix.lower()}"
            )
            _hardlink_or_copy(source, destination)
            rows.append(
                {
                    "order": image_number,
                    "arm": arm_name,
                    "view_id": item.get("view_id"),
                    "presentation": presentation,
                    "role": item.get("role"),
                    "file": destination.name,
                    "source": str(source),
                }
            )
    return rows


def _hardlink_or_copy(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"review image does not exist: {source}")
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _write_index(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "metric",
        "severity_class",
        "case_id",
        "event_id",
        "object_ids",
        "fixed_global_review",
        "metric_local_review",
        "notes",
        "event_folder",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _write_readme(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    metric_counts = {
        metric: sum(row["metric"] == metric for row in rows)
        for metric in METRIC_DIRS
    }
    text = f"""# Camera Evidence Human Review

共 {len(rows)} 个 distorted metric-events.

- Collision: {metric_counts['collision']}.
- OOB: {metric_counts['oob']}.
- Support: {metric_counts['support']}.

Folder order:

1. `00_overview`: contact sheets、HTML index 和 shared review table.
2. `01_collision`: Collision events，先 obvious 后 subtle.
3. `02_oob`: OOB events，先 obvious 后 subtle.
4. `03_support`: Support events，先 obvious 后 subtle.

每个 event folder 中图片固定按以下顺序排列:

1. `FG`: fixed global top raw + highlight.
2. `FG`: fixed global perspective raw + highlight.
3. `ML`: metric-local selected pose 1 raw + highlight.
4. `ML`: metric-local selected pose 2 raw + highlight.

请分别给两个 arms 标记 `sufficient / insufficient / unclear`. Pixel visibility 只是 selector diagnostic，不是 sufficiency label. `event_summary.json` 保存 metric-local selector、selected view IDs 和 fallback reason.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "item"


if __name__ == "__main__":
    main()
