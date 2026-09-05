#!/usr/bin/env python3
"""Aggregate two exact-input cal_dataset2 non-L1 judgement repeats."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = (
    PROJECT_ROOT
    / "Support"
    / "artifacts"
    / "outputs"
    / "exp2_non_l1_visual_evidence_gpt56"
)


def main() -> None:
    args = _parse_args()
    left_root = args.left_run.expanduser().resolve()
    right_root = args.right_run.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    left = _read_results(left_root)
    right = _read_results(right_root)
    if set(left) != set(right):
        missing_left = sorted(set(right) - set(left))
        missing_right = sorted(set(left) - set(right))
        raise ValueError(
            f"repeat job sets differ: missing_left={missing_left[:5]} "
            f"missing_right={missing_right[:5]}"
        )

    paired: list[dict[str, Any]] = []
    for key in sorted(left):
        lrow = left[key]
        rrow = right[key]
        if lrow["evidence_packet_sha256"] != rrow["evidence_packet_sha256"]:
            raise ValueError(f"evidence hash drift between repeats for {key}")
        paired.append(
            {
                "case_id": lrow["case_id"],
                "event_id": lrow["event_id"],
                "metric": lrow["metric"],
                "arm": lrow["arm"],
                "gt_label": lrow["gt_label"],
                "repeat_1_prediction": lrow["predicted_label"],
                "repeat_2_prediction": rrow["predicted_label"],
                "repeat_1_evidence_status": lrow["evidence_status"],
                "repeat_2_evidence_status": rrow["evidence_status"],
                "repeat_1_binary_match": lrow["binary_match"],
                "repeat_2_binary_match": rrow["binary_match"],
                "label_agreement": _same_nonempty(
                    lrow["predicted_label"], rrow["predicted_label"]
                ),
                "evidence_status_agreement": _same_nonempty(
                    lrow["evidence_status"], rrow["evidence_status"]
                ),
                "both_binary_correct": _both_true(
                    lrow["binary_match"], rrow["binary_match"]
                ),
                "stable_binary_error": _both_false(
                    lrow["binary_match"], rrow["binary_match"]
                ),
                "evidence_packet_sha256": lrow["evidence_packet_sha256"],
                "repeat_1_error": lrow["error"],
                "repeat_2_error": rrow["error"],
            }
        )

    summary = _summary_rows(paired, left, right)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_tsv(out_dir / "paired_repeats.tsv", paired)
    _write_tsv(out_dir / "repeatability_summary.tsv", summary)
    report = _report(summary, paired)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    manifest = {
        "schema_version": "cal_dataset2_non_l1_repeat_analysis_v1",
        "left_run": str(left_root),
        "right_run": str(right_root),
        "paired_job_count": len(paired),
        "complete_pair_count": sum(
            not row["repeat_1_error"] and not row["repeat_2_error"]
            for row in paired
        ),
        "outputs": {
            "paired_repeats": str((out_dir / "paired_repeats.tsv").resolve()),
            "summary": str((out_dir / "repeatability_summary.tsv").resolve()),
            "report": str((out_dir / "report.md").resolve()),
        },
        "source_sha256": {
            "left_per_event": _file_sha256(left_root / "per_event.tsv"),
            "right_per_event": _file_sha256(right_root / "per_event.tsv"),
        },
    }
    _write_json(out_dir / "analysis_manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


def _summary_rows(
    paired: list[dict[str, Any]],
    left: dict[tuple[str, str], dict[str, str]],
    right: dict[tuple[str, str], dict[str, str]],
) -> list[dict[str, Any]]:
    groups: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in paired:
        groups[(row["metric"], row["arm"])].append(row)
        groups[("ALL", row["arm"])].append(row)
    output = []
    for (metric, arm), rows in sorted(groups.items()):
        complete = [
            row
            for row in rows
            if not row["repeat_1_error"] and not row["repeat_2_error"]
        ]
        binary = [
            row
            for row in complete
            if row["gt_label"] in {"valid", "invalid"}
        ]
        keys = [(row["case_id"], row["arm"]) for row in complete]
        left_rows = [left[key] for key in keys]
        right_rows = [right[key] for key in keys]
        output.append(
            {
                "metric": metric,
                "arm": arm,
                "paired_n": len(rows),
                "complete_pair_n": len(complete),
                "label_repeat_agreement": _mean(
                    [row["label_agreement"] for row in complete]
                ),
                "evidence_status_repeat_agreement": _mean(
                    [row["evidence_status_agreement"] for row in complete]
                ),
                "binary_n": len(binary),
                "binary_accuracy_repeat_1": _mean(
                    [_bool_value(row["binary_match"]) for row in left_rows]
                ),
                "binary_accuracy_repeat_2": _mean(
                    [_bool_value(row["binary_match"]) for row in right_rows]
                ),
                "stable_binary_correct_rate": _mean(
                    [row["both_binary_correct"] for row in binary]
                ),
                "stable_binary_error_rate": _mean(
                    [row["stable_binary_error"] for row in binary]
                ),
                "evidence_sufficient_rate_repeat_1": _mean(
                    [
                        row["evidence_status"] == "sufficient"
                        for row in left_rows
                        if row["evidence_status"]
                    ]
                ),
                "evidence_sufficient_rate_repeat_2": _mean(
                    [
                        row["evidence_status"] == "sufficient"
                        for row in right_rows
                        if row["evidence_status"]
                    ]
                ),
                "mean_confidence_repeat_1": _mean_float(
                    [row["confidence"] for row in left_rows]
                ),
                "mean_confidence_repeat_2": _mean_float(
                    [row["confidence"] for row in right_rows]
                ),
            }
        )
    return output


def _report(
    summary: list[dict[str, Any]],
    paired: list[dict[str, Any]],
) -> str:
    all_rows = [
        row
        for row in summary
        if row["metric"] == "ALL"
    ]
    lines = [
        "# cal_dataset2 non-L1 exact-input repeat analysis",
        "",
        "This report compares two byte-identical evidence replays with the same "
        "GPT-5.6-Sol configuration. Missing calls remain unresolved.",
        "",
        "## Overall by evidence arm",
        "",
        "| Arm | Complete pairs | Label agreement | Binary acc. R1 | Binary acc. R2 | Evidence sufficient R1/R2 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in all_rows:
        lines.append(
            "| {arm} | {n} | {agree} | {a1} | {a2} | {s1} / {s2} |".format(
                arm=row["arm"],
                n=row["complete_pair_n"],
                agree=_percent(row["label_repeat_agreement"]),
                a1=_percent(row["binary_accuracy_repeat_1"]),
                a2=_percent(row["binary_accuracy_repeat_2"]),
                s1=_percent(row["evidence_sufficient_rate_repeat_1"]),
                s2=_percent(row["evidence_sufficient_rate_repeat_2"]),
            )
        )
    disagreements = [
        row
        for row in paired
        if row["label_agreement"] is False
        and not row["repeat_1_error"]
        and not row["repeat_2_error"]
    ]
    stable_errors = [
        row
        for row in paired
        if row["stable_binary_error"] is True
    ]
    lines.extend(
        [
            "",
            "## Diagnostics",
            "",
            f"- Repeat label disagreements: {len(disagreements)} / {len(paired)}.",
            f"- Stable binary errors: {len(stable_errors)}.",
            "- Metric-level and arm-level values are in `repeatability_summary.tsv`.",
            "- Per-case transitions are in `paired_repeats.tsv`.",
            "",
        ]
    )
    return "\n".join(lines)


def _read_results(root: Path) -> dict[tuple[str, str], dict[str, str]]:
    rows = _read_tsv(root / "per_event.tsv")
    output = {}
    for row in rows:
        key = (row["case_id"], row["arm"])
        if key in output:
            raise ValueError(f"duplicate per-event result: {key}")
        output[key] = row
    if not output:
        raise ValueError(f"no per-event results in {root}")
    return output


def _same_nonempty(left: str, right: str) -> bool | None:
    return left == right if left and right else None


def _both_true(left: str, right: str) -> bool | None:
    values = (_bool_value(left), _bool_value(right))
    return all(values) if all(value is not None for value in values) else None


def _both_false(left: str, right: str) -> bool | None:
    values = (_bool_value(left), _bool_value(right))
    return not any(values) if all(value is not None for value in values) else None


def _bool_value(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in {"True", "true", "1", 1}:
        return True
    if value in {"False", "false", "0", 0}:
        return False
    return None


def _mean(values: list[bool | None]) -> float | None:
    selected = [value for value in values if isinstance(value, bool)]
    return sum(selected) / len(selected) if selected else None


def _mean_float(values: list[Any]) -> float | None:
    selected = []
    for value in values:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if not parsed != parsed:
            selected.append(parsed)
    return statistics.fmean(selected) if selected else None


def _percent(value: Any) -> str:
    return "—" if value is None else f"{float(value):.1%}"


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]) if rows else [],
            delimiter="\t",
            lineterminator="\n",
        )
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-run", type=Path, default=DEFAULT_ROOT / "repeat_1")
    parser.add_argument("--right-run", type=Path, default=DEFAULT_ROOT / "repeat_2")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_ROOT / "analysis")
    return parser.parse_args()


if __name__ == "__main__":
    main()
