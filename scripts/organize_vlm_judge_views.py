#!/usr/bin/env python3
"""Organize archived VLM evidence for event-by-event visual auditing."""

from __future__ import annotations

import argparse
import csv
import os
import re
from collections import Counter, defaultdict
from pathlib import Path


METRIC_DIRS = {
    "collision": "collision",
    "oob": "oob",
    "support": "support",
}

ARM_DIRS = {
    "global_raw": "fixed_global_raw",
    "visibility_raw": "visibility_local_raw",
    "visibility_highlight": "visibility_local_highlight",
    "visibility_highlight_global": "visibility_local_highlight_plus_global",
}


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _resolve_repo_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _safe_component(value: str, *, fallback: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
    return normalized or fallback


def _image_name(row: dict[str, str], source: Path) -> str:
    index_text = row.get("image_index", "").strip()
    try:
        index = f"{int(index_text):02d}"
    except ValueError:
        index = "xx"
    role = _safe_component(row.get("role", ""), fallback="view")
    view_id = _safe_component(row.get("view_id", ""), fallback=source.stem)
    return f"{index}_{role}_{view_id}{source.suffix.lower()}"


def _prune_empty_directories(root: Path) -> None:
    if not root.exists():
        return
    for directory in sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        try:
            directory.rmdir()
        except OSError:
            pass
    try:
        root.rmdir()
    except OSError:
        pass


def _remove_legacy_support_index(root: Path) -> None:
    """Remove stale indexes from the former `floating_support` archive name."""
    if not root.exists():
        return
    if any(root.rglob("*.png")):
        raise RuntimeError(f"refusing to remove legacy directory containing images: {root}")
    for filename in ("judge_view_manifest.csv", "unresolved_events.csv", "event_index.csv"):
        path = root / filename
        if path.exists():
            path.unlink()
    _prune_empty_directories(root)


def organize(repo_root: Path, archive_root: Path) -> None:
    manifest_path = archive_root / "judge_view_manifest.csv"
    unresolved_path = archive_root / "unresolved_events.csv"
    fieldnames, rows = _read_csv(manifest_path)

    planned_moves: list[tuple[dict[str, str], Path, Path, str, str]] = []
    destinations: set[Path] = set()
    for row in rows:
        metric = row["metric"]
        arm = row["arm"]
        metric_dir = METRIC_DIRS.get(metric)
        arm_dir = ARM_DIRS.get(arm)
        if metric_dir is None or arm_dir is None:
            raise ValueError(f"unsupported metric/arm pair: {metric!r}/{arm!r}")

        source = _resolve_repo_path(repo_root, row["archive_path"])
        case_dir = _safe_component(row.get("case_id", ""), fallback="unknown_case")
        event_dir = "event_" + _safe_component(
            row.get("event_id", ""), fallback="unknown_event"
        )
        destination = (
            archive_root
            / metric_dir
            / case_dir
            / event_dir
            / arm_dir
            / _image_name(row, source)
        )
        if destination in destinations:
            raise ValueError(f"multiple manifest rows map to {destination}")
        destinations.add(destination)
        planned_moves.append((row, source, destination, metric_dir, arm_dir))

    for _, source, destination, _, _ in planned_moves:
        if source == destination:
            if not source.exists():
                raise FileNotFoundError(f"archived image is missing: {source}")
            continue
        if not source.exists():
            raise FileNotFoundError(f"archived image is missing: {source}")
        if destination.exists():
            raise FileExistsError(f"destination already exists: {destination}")

    by_metric: dict[str, list[dict[str, str]]] = defaultdict(list)
    counts: Counter[tuple[str, str]] = Counter()
    event_rows: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    moved = 0
    for row, source, destination, metric_dir, arm_dir in planned_moves:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source != destination:
            os.replace(source, destination)
            moved += 1
        row["archive_path"] = destination.relative_to(repo_root).as_posix()
        by_metric[metric_dir].append(row)
        counts[(metric_dir, arm_dir)] += 1
        event_rows[(metric_dir, row["case_id"], row["event_id"])].append(row)

    _write_csv(manifest_path, fieldnames, rows)
    for metric_dir, metric_rows in sorted(by_metric.items()):
        _write_csv(archive_root / metric_dir / "judge_view_manifest.csv", fieldnames, metric_rows)

    event_index_fields = [
        "metric",
        "case_id",
        "event_id",
        "object_ids",
        "gt_label",
        "visual_configs",
        "image_count",
    ]
    event_index_by_metric: dict[str, list[dict[str, str]]] = defaultdict(list)
    for (metric_dir, case_id, event_id), grouped_rows in sorted(event_rows.items()):
        first = grouped_rows[0]
        event_index_by_metric[metric_dir].append(
            {
                "metric": first["metric"],
                "case_id": case_id,
                "event_id": event_id,
                "object_ids": first.get("object_ids", ""),
                "gt_label": first.get("gt_label", ""),
                "visual_configs": ",".join(sorted({row["arm"] for row in grouped_rows})),
                "image_count": str(len(grouped_rows)),
            }
        )
    for metric_dir, index_rows in sorted(event_index_by_metric.items()):
        _write_csv(
            archive_root / metric_dir / "event_index.csv",
            event_index_fields,
            index_rows,
        )

    unresolved_by_metric: dict[str, list[dict[str, str]]] = defaultdict(list)
    if unresolved_path.exists():
        unresolved_fields, unresolved_rows = _read_csv(unresolved_path)
        for row in unresolved_rows:
            metric_dir = METRIC_DIRS.get(row["metric"])
            if metric_dir is None:
                raise ValueError(f"unsupported unresolved metric: {row['metric']!r}")
            unresolved_by_metric[metric_dir].append(row)
        for metric_dir in sorted(set(by_metric) | set(unresolved_by_metric)):
            _write_csv(
                archive_root / metric_dir / "unresolved_events.csv",
                unresolved_fields,
                unresolved_by_metric.get(metric_dir, []),
            )

    # Remove only empty directories left by the former config-first hierarchy.
    _remove_legacy_support_index(archive_root / "floating_support")
    for metric_dir in METRIC_DIRS.values():
        metric_root = archive_root / metric_dir
        for old_arm_dir in ARM_DIRS.values():
            _prune_empty_directories(metric_root / old_arm_dir)

    oar_dir = archive_root / "oar"
    oar_dir.mkdir(parents=True, exist_ok=True)
    (oar_dir / "README.md").write_text(
        "# OAR Judge Views\n\n"
        "This controlled P0b archive contains no dedicated OAR visual-judge packets. "
        "Cases whose IDs end in `__oar` are distortion-family fixtures; their archived "
        "images were consumed by Collision, OOB, or Support checks and remain filed "
        "under the actual judge metric.\n",
        encoding="utf-8",
    )

    summary_lines = [
        "# VLM Judge View Archive",
        "",
        "The archive is case-first within each metric so every physical event can be "
        "compared across visual configurations without searching long filenames.",
        "Original experiment files under `result/benchmark_metric_analysis/pictures/` are unchanged.",
        "",
        "```text",
        "vlm_judge_views/",
        "  <metric>/",
        "    <case_id>/",
        "      event_<event_id>/",
        "        <visual-config>/",
        "          <image-index>_<role>_<view-id>.png",
        "```",
        "",
        "Metrics with archived packets: `collision`, `oob`, and `support`. The `oar/` "
        "directory explains why this experiment has no dedicated OAR packets.",
        "",
        "## Visual Configurations",
        "",
        "- `fixed_global_raw`: fixed overview renders without a local focus view.",
        "- `visibility_local_raw`: deterministic local candidates selected for visibility/framing.",
        "- `visibility_local_highlight`: the same local policy with target highlighting.",
        "- `visibility_local_highlight_plus_global`: highlighted local evidence plus global context.",
        "",
        "## Image Counts",
        "",
        "| Metric | Visual config | Images |",
        "| --- | --- | ---: |",
    ]
    for (metric_dir, arm_dir), count in sorted(counts.items()):
        summary_lines.append(f"| {metric_dir} | {arm_dir} | {count} |")
    summary_lines.extend(
        [
            "",
            f"Organized {len(rows)} archived image entries; {moved} moved in this run.",
            "The root `judge_view_manifest.csv` is authoritative. Each metric also has "
            "a filtered manifest and an `event_index.csv`.",
        ]
    )
    (archive_root / "README.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=Path("Support/artifacts/result/vlm_judge_views"),
    )
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    archive_root = args.archive_root
    if not archive_root.is_absolute():
        archive_root = repo_root / archive_root
    organize(repo_root, archive_root.resolve())


if __name__ == "__main__":
    main()
