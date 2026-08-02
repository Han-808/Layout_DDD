#!/usr/bin/env python3
"""Merge scored invalid Job 1 results with repeated ambiguous-edge results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


KEY_FIELDS = ("case_id", "metric", "event_id", "arm")
ARMS = ("fixed_global_highlight", "metric_local_highlight")
METRICS = ("collision", "oob", "support")


def main() -> None:
    args = _parse_args()
    invalid_dir = Path(args.invalid_run).expanduser().resolve()
    ambiguous_dirs = [
        Path(value).expanduser().resolve() for value in args.ambiguous_run
    ]
    invalid_repeat_path = (
        Path(args.invalid_repeat_summary).expanduser().resolve()
        if args.invalid_repeat_summary
        else None
    )
    out_dir = Path(args.out_dir).expanduser().resolve()
    invalid = _read_tsv(invalid_dir / "per_event.tsv")
    ambiguous_runs = [
        _read_tsv(path / "per_event.tsv") for path in ambiguous_dirs
    ]
    _validate_labels(invalid, expected="invalid", name="invalid run")
    for index, rows in enumerate(ambiguous_runs, start=1):
        _validate_labels(rows, expected="ambiguous", name=f"ambiguous run {index}")
    invalid_repeat_rows = (
        _read_tsv(invalid_repeat_path)
        if invalid_repeat_path and invalid_repeat_path.is_file()
        else []
    )

    master: list[dict[str, Any]] = []
    for row in invalid:
        master.append({"stratum": "invalid", "repeat": 1, **row})
    for repeat, rows in enumerate(ambiguous_runs, start=1):
        for row in rows:
            master.append({"stratum": "ambiguous", "repeat": repeat, **row})

    summaries = _summary_rows(invalid, ambiguous_runs, invalid_repeat_rows)
    unique_events = {
        (row["case_id"], row["metric"], row["event_id"])
        for row in master
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_tsv(out_dir / "master_results.tsv", master)
    _write_tsv(out_dir / "summary.tsv", summaries)
    prior_combined = next(
        (
            row
            for row in invalid_repeat_rows
            if row.get("comparison") == "combined_exact_input_repeat"
        ),
        None,
    )
    ambiguous_overall = [
        row
        for row in summaries
        if row["group_type"] == "overall" and row["group"] == "overall"
    ]
    ambiguous_repeat_pairs = sum(
        int(row["ambiguous_exact_input_repeat_pairs"])
        for row in ambiguous_overall
    )
    ambiguous_repeat_agreements = sum(
        round(
            int(row["ambiguous_exact_input_repeat_pairs"])
            * float(row["ambiguous_repeat_agreement"])
        )
        for row in ambiguous_overall
        if row["ambiguous_repeat_agreement"] is not None
    )
    prior_repeat_pairs = int(prior_combined["pairs"]) if prior_combined else 0
    prior_repeat_agreements = (
        int(prior_combined["agreements"]) if prior_combined else 0
    )
    pooled_repeat_pairs = prior_repeat_pairs + ambiguous_repeat_pairs
    pooled_repeat_agreements = (
        prior_repeat_agreements + ambiguous_repeat_agreements
    )
    manifest = {
        "schema_version": "cal_dataset1_job1_extended_merge_v1",
        "invalid_run": str(invalid_dir),
        "ambiguous_runs": [str(path) for path in ambiguous_dirs],
        "source_per_event_sha256": {
            str(invalid_dir): _file_sha256(invalid_dir / "per_event.tsv"),
            **{
                str(path): _file_sha256(path / "per_event.tsv")
                for path in ambiguous_dirs
            },
        },
        "invalid_repeat_summary": (
            str(invalid_repeat_path)
            if invalid_repeat_path and invalid_repeat_path.is_file()
            else None
        ),
        "invalid_repeat_summary_sha256": (
            _file_sha256(invalid_repeat_path)
            if invalid_repeat_path and invalid_repeat_path.is_file()
            else None
        ),
        "unique_event_count": len(unique_events),
        "invalid_event_count": len(
            {
                (row["case_id"], row["metric"], row["event_id"])
                for row in invalid
            }
        ),
        "ambiguous_event_count": len(
            {
                (row["case_id"], row["metric"], row["event_id"])
                for row in ambiguous_runs[0]
            }
        ),
        "repeatability": {
            "prior_invalid_exact_input": {
                "pairs": prior_repeat_pairs,
                "agreements": prior_repeat_agreements,
                "agreement_rate": (
                    prior_repeat_agreements / prior_repeat_pairs
                    if prior_repeat_pairs
                    else None
                ),
            },
            "ambiguous_exact_input": {
                "pairs": ambiguous_repeat_pairs,
                "agreements": ambiguous_repeat_agreements,
                "agreement_rate": (
                    ambiguous_repeat_agreements / ambiguous_repeat_pairs
                    if ambiguous_repeat_pairs
                    else None
                ),
            },
            "pooled_exact_input": {
                "pairs": pooled_repeat_pairs,
                "agreements": pooled_repeat_agreements,
                "agreement_rate": (
                    pooled_repeat_agreements / pooled_repeat_pairs
                    if pooled_repeat_pairs
                    else None
                ),
            },
        },
        "interpretation": (
            "Invalid recall is computed from the invalid stratum only. "
            "Ambiguous events contribute verdict tendency and repeatability, "
            "not binary accuracy."
        ),
        "outputs": {
            "master_results": str((out_dir / "master_results.tsv").resolve()),
            "summary": str((out_dir / "summary.tsv").resolve()),
        },
    }
    _write_json(out_dir / "run_manifest.json", manifest)
    _write_json(
        out_dir / "summary.json",
        {"manifest": manifest, "summary": summaries},
    )
    print(json.dumps(manifest, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--invalid-run",
        default="Support/artifacts/outputs/exp1_1_gpt56_judge",
    )
    parser.add_argument("--ambiguous-run", action="append", required=True)
    parser.add_argument(
        "--invalid-repeat-summary",
        default=(
            "Support/artifacts/visualizations/"
            "visual_config_gpt56_20260722/repeatability.tsv"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="Support/artifacts/outputs/exp1_1_extended_gpt56",
    )
    return parser.parse_args()


def _summary_rows(
    invalid: list[dict[str, str]],
    ambiguous_runs: list[list[dict[str, str]]],
    invalid_repeat_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        groups = [("overall", "overall")]
        groups.extend(("metric", metric) for metric in METRICS)
        for group_type, group in groups:
            invalid_selected = _select(invalid, arm, group_type, group)
            ambiguous_selected = [
                _select(run, arm, group_type, group) for run in ambiguous_runs
            ]
            if not invalid_selected and not any(ambiguous_selected):
                continue
            invalid_detected = sum(
                row["predicted_label"] == "invalid" for row in invalid_selected
            )
            ambiguous_flat = [
                row for selected in ambiguous_selected for row in selected
            ]
            ambiguous_invalid = sum(
                row["predicted_label"] == "invalid" for row in ambiguous_flat
            )
            repeat_exact = 0
            repeat_agreed = 0
            if len(ambiguous_selected) >= 2:
                left = _index(ambiguous_selected[0])
                right = _index(ambiguous_selected[1])
                if set(left) != set(right):
                    raise RuntimeError(
                        f"ambiguous repeat keys differ for {arm} {group}"
                    )
                for key in left:
                    first = left[key]
                    second = right[key]
                    if (
                        first["contract_sha256"] == second["contract_sha256"]
                        and first["comparison_manifest_sha256"]
                        == second["comparison_manifest_sha256"]
                    ):
                        repeat_exact += 1
                        repeat_agreed += (
                            first["predicted_label"]
                            == second["predicted_label"]
                        )
            unique_ambiguous = len(ambiguous_selected[0]) if ambiguous_selected else 0
            prior_repeat = _prior_repeat_for_arm(
                invalid_repeat_rows,
                arm=arm,
                group_type=group_type,
            )
            prior_repeat_pairs = (
                int(prior_repeat["pairs"]) if prior_repeat else 0
            )
            prior_repeat_agreements = (
                int(prior_repeat["agreements"]) if prior_repeat else 0
            )
            ambiguous_repeat_agreements = (
                round(repeat_agreed) if repeat_exact else 0
            )
            combined_repeat_pairs = prior_repeat_pairs + repeat_exact
            combined_repeat_agreements = (
                prior_repeat_agreements + ambiguous_repeat_agreements
            )
            rows.append(
                {
                    "arm": arm,
                    "group_type": group_type,
                    "group": group,
                    "invalid_events": len(invalid_selected),
                    "invalid_detected": invalid_detected,
                    "invalid_recall": (
                        invalid_detected / len(invalid_selected)
                        if invalid_selected
                        else None
                    ),
                    "ambiguous_events": unique_ambiguous,
                    "ambiguous_repeat_count": len(ambiguous_runs),
                    "ambiguous_judgements": len(ambiguous_flat),
                    "ambiguous_predicted_invalid": ambiguous_invalid,
                    "ambiguous_invalid_rate": (
                        ambiguous_invalid / len(ambiguous_flat)
                        if ambiguous_flat
                        else None
                    ),
                    "ambiguous_exact_input_repeat_pairs": repeat_exact,
                    "ambiguous_repeat_agreement": (
                        repeat_agreed / repeat_exact if repeat_exact else None
                    ),
                    "prior_invalid_exact_input_repeat_pairs": (
                        prior_repeat_pairs if prior_repeat else None
                    ),
                    "prior_invalid_repeat_agreements": (
                        prior_repeat_agreements if prior_repeat else None
                    ),
                    "combined_exact_input_repeat_pairs": (
                        combined_repeat_pairs
                        if group_type == "overall" and prior_repeat
                        else None
                    ),
                    "combined_repeat_agreements": (
                        combined_repeat_agreements
                        if group_type == "overall" and prior_repeat
                        else None
                    ),
                    "combined_repeat_agreement": (
                        combined_repeat_agreements / combined_repeat_pairs
                        if (
                            group_type == "overall"
                            and prior_repeat
                            and combined_repeat_pairs
                        )
                        else None
                    ),
                    "total_unique_events": (
                        len(invalid_selected) + unique_ambiguous
                    ),
                }
            )
    return rows


def _prior_repeat_for_arm(
    rows: list[dict[str, str]],
    *,
    arm: str,
    group_type: str,
) -> dict[str, str] | None:
    if group_type != "overall":
        return None
    comparison = {
        "fixed_global_highlight": "fixed_global_highlight_repeat",
        "metric_local_highlight": "local_raw_highlight_repeat",
    }[arm]
    return next(
        (row for row in rows if row.get("comparison") == comparison),
        None,
    )


def _select(
    rows: list[dict[str, str]],
    arm: str,
    group_type: str,
    group: str,
) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row["arm"] == arm
        and (group_type == "overall" or row["metric"] == group)
    ]


def _index(
    rows: list[dict[str, str]],
) -> dict[tuple[str, ...], dict[str, str]]:
    result: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        key = tuple(row[field] for field in KEY_FIELDS)
        if key in result:
            raise ValueError(f"duplicate result key: {key}")
        result[key] = row
    return result


def _validate_labels(
    rows: list[dict[str, str]],
    *,
    expected: str,
    name: str,
) -> None:
    labels = {row.get("gt_label") for row in rows}
    if labels != {expected}:
        raise ValueError(f"{name} expected GT={expected}, found {sorted(labels)}")


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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
