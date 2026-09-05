#!/usr/bin/env python3
"""Compare two exact-input cal_dataset1 judge runs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


KEY_FIELDS = ("case_id", "metric", "event_id", "arm")


def main() -> None:
    args = _parse_args()
    left_dir = Path(args.left_run).expanduser().resolve()
    right_dir = Path(args.right_run).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    left = _index(_read_tsv(left_dir / "per_event.tsv"))
    right = _index(_read_tsv(right_dir / "per_event.tsv"))
    if set(left) != set(right):
        missing_left = sorted(set(right) - set(left))
        missing_right = sorted(set(left) - set(right))
        raise RuntimeError(
            "repeat key sets differ: "
            f"missing_left={missing_left[:3]} missing_right={missing_right[:3]}"
        )

    rows: list[dict[str, Any]] = []
    for key in sorted(left):
        first = left[key]
        second = right[key]
        exact_input = bool(
            first.get("contract_sha256")
            and first.get("contract_sha256") == second.get("contract_sha256")
            and first.get("comparison_manifest_sha256")
            == second.get("comparison_manifest_sha256")
        )
        rows.append(
            {
                **dict(zip(KEY_FIELDS, key, strict=True)),
                "severity_class": first.get("severity_class"),
                "gt_label": first.get("gt_label"),
                "exact_input": int(exact_input),
                "left_prediction": first.get("predicted_label"),
                "right_prediction": second.get("predicted_label"),
                "verdict_agreement": int(
                    first.get("predicted_label") == second.get("predicted_label")
                ),
                "left_confidence": first.get("confidence"),
                "right_confidence": second.get("confidence"),
                "left_contract_sha256": first.get("contract_sha256"),
                "right_contract_sha256": second.get("contract_sha256"),
            }
        )

    summaries = _summary_rows(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_tsv(out_dir / "per_event_repeatability.tsv", rows)
    _write_tsv(out_dir / "summary.tsv", summaries)
    _write_json(
        out_dir / "summary.json",
        {
            "left_run": str(left_dir),
            "right_run": str(right_dir),
            "summary": summaries,
        },
    )
    overall = next(
        row
        for row in summaries
        if row["group_type"] == "overall" and row["group"] == "overall"
    )
    print(json.dumps(overall, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-run", required=True)
    parser.add_argument("--right-run", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def _index(rows: list[dict[str, str]]) -> dict[tuple[str, ...], dict[str, str]]:
    result: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        key = tuple(row.get(field, "") for field in KEY_FIELDS)
        if key in result:
            raise ValueError(f"duplicate repeat key: {key}")
        result[key] = row
    return result


def _summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: list[tuple[str, str, list[dict[str, Any]]]] = [
        ("overall", "overall", rows)
    ]
    groups.extend(
        ("metric", metric, [row for row in rows if row["metric"] == metric])
        for metric in ("collision", "oob", "support")
    )
    groups.extend(
        ("arm", arm, [row for row in rows if row["arm"] == arm])
        for arm in sorted({str(row["arm"]) for row in rows})
    )
    summaries: list[dict[str, Any]] = []
    for group_type, group, selected in groups:
        if not selected:
            continue
        exact = [row for row in selected if row["exact_input"]]
        agreed = [row for row in exact if row["verdict_agreement"]]
        summaries.append(
            {
                "group_type": group_type,
                "group": group,
                "total_pairs": len(selected),
                "exact_input_pairs": len(exact),
                "agreement_count": len(agreed),
                "agreement_rate": len(agreed) / len(exact) if exact else None,
                "invalid_to_valid": sum(
                    row["left_prediction"] == "invalid"
                    and row["right_prediction"] == "valid"
                    for row in exact
                ),
                "valid_to_invalid": sum(
                    row["left_prediction"] == "valid"
                    and row["right_prediction"] == "invalid"
                    for row in exact
                ),
            }
        )
    return summaries


def _read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty TSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
