#!/usr/bin/env python3
"""Create a human-review bundle for the candidate Scene10 FrozenAssets set."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    spec = _read(args.spec)
    rows = _rows(spec, args.asset_root.expanduser().resolve())
    groups = _group_rows(rows)
    summary = _summary(spec, rows, groups)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        args.out_dir / "asset_review.json",
        {"summary": summary, "groups": groups, "rows": rows},
    )
    _write_csv(args.out_dir / "asset_review.csv", rows)
    _write_csv(args.out_dir / "group_review.csv", groups)
    (args.out_dir / "README.md").write_text(
        _markdown(summary, groups), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "out_dir": args.out_dir.resolve().as_posix(),
                "slots": len(rows),
                "assets": summary["unique_assets"],
                "flagged_slots": summary["flagged_slots"],
            },
            sort_keys=True,
        )
    )


def _rows(spec: dict[str, Any], asset_root: Path) -> list[dict[str, Any]]:
    uses: dict[str, set[str]] = defaultdict(set)
    counts: Counter[str] = Counter()
    for case in spec["cases"]:
        for item in case["objects"]:
            asset_id = str(item["asset_id"])
            counts[asset_id] += 1
            uses[asset_id].add(str(item.get("metadata", {}).get("requested_category")))
    result = []
    for case in spec["cases"]:
        room = case["room"]
        xs = [float(point[0]) for point in room["boundary"]]
        ys = [float(point[1]) for point in room["boundary"]]
        room_size = [max(xs) - min(xs), max(ys) - min(ys), float(room["height"])]
        public = {item["id"]: item for item in case["object_plan"]["objects"]}
        public_ids = set(public)
        catalog = {item["asset_id"]: item for item in spec["catalog"]["assets"]}
        for frozen in case["objects"]:
            slot_id = str(frozen["slot_id"])
            asset_id = str(frozen["asset_id"])
            public_item = public[slot_id]
            metadata_path = asset_root / asset_id / f"{asset_id}_metadata.json"
            metadata = _read(metadata_path)
            size = [float(value) for value in metadata["transformed_size"]]
            center = [float(value) for value in metadata["transformed_bbox_center"]]
            estimated = public_item.get("estimated_size")
            ratios = (
                [
                    size[index] / float(estimated[index])
                    for index in range(3)
                    if float(estimated[index]) > 0.0
                ]
                if isinstance(estimated, list) and len(estimated) == 3
                else []
            )
            directed = bool(public_item.get("metadata", {}).get("directed"))
            asset = catalog[asset_id]
            requested_category = str(public_item["category"])
            asset_category = str(asset["category"])
            flags = []
            if not _category_overlap(requested_category, asset_category):
                flags.append("semantic_category_review")
            if ratios and (min(ratios) < 0.5 or max(ratios) > 2.0):
                flags.append("estimated_size_ratio_outlier")
            if size[0] > room_size[0] * 0.5 or size[1] > room_size[1] * 0.5:
                flags.append("large_room_footprint")
            if size[2] > room_size[2]:
                flags.append("exceeds_room_height")
            if directed and asset.get("canonical_front") is None:
                flags.append("directed_asset_front_unavailable")
            if len(uses[asset_id]) > 1:
                flags.append("asset_reused_across_requested_categories")
            support = str(public_item.get("metadata", {}).get("support") or "floor")
            support_kind = str(
                public_item.get("metadata", {}).get("support_kind") or (
                    "object" if support in public_ids else support
                )
            )
            support_parent = public_item.get("metadata", {}).get(
                "support_parent_id"
            )
            if support_kind == "object" and (
                not support_parent or str(support_parent) not in public_ids
            ):
                flags.append("invalid_support_parent")
            result.append(
                {
                    "case_id": case["case_id"],
                    "slot_id": slot_id,
                    "source_group_id": frozen.get("metadata", {}).get("source_group_id"),
                    "requested_category": requested_category,
                    "requested_description": public_item["description"],
                    "asset_id": asset_id,
                    "asset_category": asset_category,
                    "asset_description": asset["description"],
                    "bbox_size_m": size,
                    "room_size_m": room_size,
                    "room_height_m": room_size[2],
                    "bbox_center_local_m": center,
                    "estimated_size_m": estimated,
                    "estimated_to_asset_axis_ratios": ratios,
                    "canonical_front": asset.get("canonical_front"),
                    "directed": directed,
                    "support": support,
                    "support_kind": support_kind,
                    "support_parent_id": support_parent,
                    "global_asset_use_count": counts[asset_id],
                    "requested_categories_for_asset": sorted(uses[asset_id]),
                    "review_flags": flags,
                    "review_priority": _review_priority(flags),
                    "recommended_action": (
                        "REVIEW" if flags else "KEEP_CANDIDATE"
                    ),
                }
            )
    return result


def _group_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse count-expanded slots that share one source request and asset."""

    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row["case_id"]),
            str(row["source_group_id"]),
            str(row["asset_id"]),
        )
        current = grouped.get(key)
        if current is None:
            current = {
                "case_id": row["case_id"],
                "source_group_id": row["source_group_id"],
                "slot_ids": [],
                "instance_count": 0,
                "requested_category": row["requested_category"],
                "requested_description": row["requested_description"],
                "asset_id": row["asset_id"],
                "asset_category": row["asset_category"],
                "asset_description": row["asset_description"],
                "bbox_size_m": row["bbox_size_m"],
                "room_size_m": row["room_size_m"],
                "room_height_m": row["room_height_m"],
                "estimated_size_m": row["estimated_size_m"],
                "canonical_front": row["canonical_front"],
                "review_flags": row["review_flags"],
                "review_priority": row["review_priority"],
                "recommended_action": row["recommended_action"],
            }
            grouped[key] = current
        current["slot_ids"].append(row["slot_id"])
        current["instance_count"] += 1
    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "NONE": 3}
    return sorted(
        grouped.values(),
        key=lambda row: (
            order[row["review_priority"]],
            row["case_id"],
            row["source_group_id"],
            row["asset_id"],
        ),
    )


def _review_priority(flags: list[str]) -> str:
    selected = set(flags)
    if selected & {"large_room_footprint", "exceeds_room_height"} or {
        "semantic_category_review",
        "estimated_size_ratio_outlier",
    }.issubset(selected):
        return "HIGH"
    if selected & {
        "semantic_category_review",
        "estimated_size_ratio_outlier",
        "directed_asset_front_unavailable",
    }:
        return "MEDIUM"
    if selected:
        return "LOW"
    return "NONE"


def _summary(
    spec: dict[str, Any],
    rows: list[dict[str, Any]],
    groups: list[dict[str, Any]],
) -> dict[str, Any]:
    by_flag = Counter(flag for row in rows for flag in row["review_flags"])
    slots_by_priority = Counter(row["review_priority"] for row in rows)
    groups_by_priority = Counter(row["review_priority"] for row in groups)
    by_case = {}
    for case in spec["cases"]:
        selected = [row for row in rows if row["case_id"] == case["case_id"]]
        by_case[case["case_id"]] = {
            "slots": len(selected),
            "unique_assets": len({row["asset_id"] for row in selected}),
            "flagged_slots": sum(bool(row["review_flags"]) for row in selected),
        }
    return {
        "schema_version": "frozen_imaginarium_asset_review_v1",
        "selection_status": spec.get(
            "asset_selection_status", "candidate_pending_human_approval"
        ),
        "pilot_id": spec["pilot_id"],
        "cases": len(spec["cases"]),
        "slots": len(rows),
        "unique_assets": len({row["asset_id"] for row in rows}),
        "flagged_slots": sum(bool(row["review_flags"]) for row in rows),
        "object_groups": len(groups),
        "flagged_object_groups": sum(
            row["review_priority"] != "NONE" for row in groups
        ),
        "slots_by_priority": dict(sorted(slots_by_priority.items())),
        "object_groups_by_priority": dict(sorted(groups_by_priority.items())),
        "review_flag_counts": dict(sorted(by_flag.items())),
        "by_case": by_case,
        "automatic_flags_are_not_replacements": True,
    }


def _markdown(summary: dict[str, Any], groups: list[dict[str, Any]]) -> str:
    lines = [
        "# Frozen Imaginarium Scene10 asset review",
        "",
        f"Selection status: **{summary['selection_status']}**. "
        "No automatic replacement is made.",
        "",
        f"- Cases: {summary['cases']}",
        f"- Expanded slots: {summary['slots']}",
        f"- Source object groups: {summary['object_groups']}",
        f"- Unique assets: {summary['unique_assets']}",
        f"- Slots with at least one review flag: {summary['flagged_slots']}",
        "- Object groups by priority: "
        + ", ".join(
            f"{key}={value}"
            for key, value in summary["object_groups_by_priority"].items()
        ),
        "",
        "## Per-case counts",
        "",
        "| Case | Slots | Unique assets | Flagged slots |",
        "| --- | ---: | ---: | ---: |",
    ]
    for case_id, item in summary["by_case"].items():
        lines.append(
            f"| {case_id} | {item['slots']} | {item['unique_assets']} | "
            f"{item['flagged_slots']} |"
        )
    lines.extend(
        [
            "",
            "## High-priority shortlist",
            "",
            "HIGH means either both semantic/scale heuristics fired, the asset "
            "occupies more than half of a room axis, or its bbox exceeds the room "
            "height. These are review prompts, not "
            "automatic semantic judgments or replacements.",
            "",
            "| Case / object group | Instances | Requested | Frozen asset | Size m | Flags |",
            "| --- | ---: | --- | --- | --- | --- |",
        ]
    )
    for row in groups:
        if row["review_priority"] != "HIGH":
            continue
        size = " × ".join(f"{value:.2f}" for value in row["bbox_size_m"])
        lines.append(
            f"| {row['case_id']} / `{row['source_group_id']}` | "
            f"{row['instance_count']} | "
            f"{row['requested_category']} | `{row['asset_id']}` / "
            f"{row['asset_category']} | {size} | {', '.join(row['review_flags'])} |"
        )
    lines.extend(
        [
            "",
            "All object groups, including MEDIUM/LOW prompts, are in `group_review.csv`. "
            f"The full {summary['slots']}-slot table is in `asset_review.csv`; exact structured data is "
            "in `asset_review.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def _category_overlap(left: str, right: str) -> bool:
    aliases = {
        "sofa": "chair",
        "armchair": "chair",
        "nightstand": "table",
        "desk": "table",
        "bookcase": "shelf",
        "bookshelf": "shelf",
        "television": "tv",
    }
    left_tokens = _tokens(left, aliases)
    right_tokens = _tokens(right, aliases)
    return bool(left_tokens & right_tokens)


def _tokens(value: str, aliases: dict[str, str]) -> set[str]:
    tokens = set(re.findall(r"[a-z0-9]+", value.casefold().replace("_", " ")))
    return tokens | {aliases[token] for token in tokens if token in aliases}


def _read(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (list, dict))
                    else value
                    for key, value in row.items()
                }
            )


if __name__ == "__main__":
    main()
