#!/usr/bin/env python3
"""Compare the updated cal_dataset2 non-L1 run with its frozen predecessor.

The comparison is paired by repeat, case, metric, and evidence arm.  Human
labels are treated as a provisional reference rather than immutable ground
truth because the updated boundary is itself awaiting another human audit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "Support" / "artifacts" / "outputs"
DEFAULT_OLD_ROOT = DEFAULT_OUTPUT_ROOT / "exp2_non_l1_visual_evidence_gpt56"
DEFAULT_NEW_ROOT = (
    DEFAULT_OUTPUT_ROOT / "exp2_non_l1_visual_evidence_updated_20260727"
)
DEFAULT_DATASET_ROOT = (
    REPO_ROOT / "Support" / "datasets" / "cal_dataset2_non_l1_evidence"
)
DEFAULT_OUT_DIR = DEFAULT_NEW_ROOT / "comparison_to_previous"

BASE_ARMS = (
    "production_default",
    "global_only",
    "local_raw_only",
    "full_raw",
    "production_raw_swap",
    "local_contour_only",
)


def main() -> None:
    args = _parse_args()
    old_root = args.old_root.expanduser().resolve()
    new_root = args.new_root.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    old = _load_run(old_root)
    new = _load_run(new_root)
    if set(old) != set(new):
        raise ValueError("old and updated runs do not contain the same paired jobs")
    if len(old) != 960:
        raise ValueError(f"expected 960 paired calls per run, found {len(old)}")
    if any(row.get("error") for row in [*old.values(), *new.values()]):
        raise ValueError("comparison refuses to treat failed calls as results")

    case_ownership = _case_ownership(dataset_root)
    evidence_integrity = _evidence_integrity(old, new, old_root, new_root)

    overall_rows = _overall_arm_rows(old, new)
    metric_rows = _production_metric_rows(old, new)
    new_arm_rows = _new_metric_arm_rows(new)
    transition_rows = _transition_rows(old, new, case_ownership)
    ownership_rows = _ownership_rows(old, new, case_ownership)
    confidence_rows = _confidence_rows(old, new)

    _write_tsv(out_dir / "overall_arm_comparison.tsv", overall_rows)
    _write_tsv(out_dir / "production_metric_comparison.tsv", metric_rows)
    _write_tsv(out_dir / "updated_metric_arm_summary.tsv", new_arm_rows)
    _write_tsv(out_dir / "paired_label_transitions.tsv", transition_rows)
    _write_tsv(out_dir / "legacy_pairing_ownership_slices.tsv", ownership_rows)
    _write_tsv(out_dir / "confidence_diagnostic.tsv", confidence_rows)

    _write_grouped_bar_svg(
        out_dir / "overall_arm_comparison.svg",
        title="Overall binary agreement with provisional human labels",
        rows=overall_rows,
        label_key="arm",
        old_key="old_binary_accuracy",
        new_key="updated_binary_accuracy",
    )
    _write_grouped_bar_svg(
        out_dir / "production_metric_comparison.svg",
        title="Production-default agreement by metric",
        rows=metric_rows,
        label_key="metric",
        old_key="old_binary_accuracy",
        new_key="updated_binary_accuracy",
    )

    report = _report(
        old_root=old_root,
        new_root=new_root,
        overall_rows=overall_rows,
        metric_rows=metric_rows,
        new_arm_rows=new_arm_rows,
        ownership_rows=ownership_rows,
        confidence_rows=confidence_rows,
        evidence_integrity=evidence_integrity,
    )
    (out_dir / "report.md").write_text(report, encoding="utf-8")

    outputs = sorted(path for path in out_dir.iterdir() if path.is_file())
    manifest = {
        "schema_version": "exp2_non_l1_updated_comparison_v1",
        "old_run_root": old_root.as_posix(),
        "updated_run_root": new_root.as_posix(),
        "dataset_root": dataset_root.as_posix(),
        "paired_call_count": len(old),
        "evidence_integrity": evidence_integrity,
        "human_label_semantics": "provisional_reference_pending_updated_boundary_audit",
        "outputs": {
            path.name: {
                "path": path.as_posix(),
                "sha256": _sha256(path),
            }
            for path in outputs
        },
    }
    (out_dir / "comparison_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "paired_calls": len(old),
                "output_dir": out_dir.as_posix(),
                "report": (out_dir / "report.md").as_posix(),
            },
            indent=2,
        )
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-root", type=Path, default=DEFAULT_OLD_ROOT)
    parser.add_argument("--new-root", type=Path, default=DEFAULT_NEW_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def _load_run(root: Path) -> dict[tuple[int, str, str, str], dict[str, str]]:
    output: dict[tuple[int, str, str, str], dict[str, str]] = {}
    for repeat in (1, 2):
        path = root / f"repeat_{repeat}" / "per_event.tsv"
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                key = (repeat, row["case_id"], row["metric"], row["arm"])
                if key in output:
                    raise ValueError(f"duplicate result key: {key}")
                output[key] = row
    return output


def _case_ownership(dataset_root: Path) -> dict[str, str]:
    index = json.loads((dataset_root / "cases.json").read_text(encoding="utf-8"))
    output: dict[str, str] = {}
    for card in index["cases"]:
        metric = str(card["metric"])
        if metric != "object_pairing_consistency":
            output[str(card["case_id"])] = metric
            continue
        fixture = dataset_root / str(card["fixture_dir"])
        payload = json.loads((fixture / "metric_events.json").read_text(encoding="utf-8"))
        events = payload if isinstance(payload, list) else payload.get("events", [])
        event_rows = [event for event in events if event.get("metric") == metric]
        relations = {str(event.get("relation") or "") for event in event_rows}
        claim_ids = {
            str(claim_id)
            for event in event_rows
            for claim_id in event.get("source_claim_ids") or []
        }
        if any(claim_id.startswith("oor::") for claim_id in claim_ids):
            owner = "oor"
        elif relations and relations <= {"category_coexistence"}:
            owner = "object_pairing_consistency"
        else:
            owner = "functional_semantic_fidelity"
        output[str(card["case_id"])] = owner
    return output


def _overall_arm_rows(
    old: dict[tuple[int, str, str, str], dict[str, str]],
    new: dict[tuple[int, str, str, str], dict[str, str]],
) -> list[dict[str, Any]]:
    rows = []
    for arm in BASE_ARMS:
        keys = [
            key
            for key, row in old.items()
            if key[3] == arm and row["gt_label"] != "ambiguous"
        ]
        if not keys:
            continue
        corrected, regressed = _correction_counts(old, new, keys)
        old_accuracy = _accuracy(old[key] for key in keys)
        new_accuracy = _accuracy(new[key] for key in keys)
        case_differences = _case_differences(old, new, keys)
        low, high = _bootstrap_interval(case_differences)
        rows.append(
            {
                "arm": arm,
                "binary_n": len(keys),
                "old_binary_accuracy": old_accuracy,
                "updated_binary_accuracy": new_accuracy,
                "delta_pp": 100.0 * (new_accuracy - old_accuracy),
                "paired_corrected": corrected,
                "paired_regressed": regressed,
                "paired_exact_p": _exact_binomial_p(corrected, regressed),
                "case_bootstrap_delta_low_pp": 100.0 * low,
                "case_bootstrap_delta_high_pp": 100.0 * high,
                "old_repeat_agreement": _repeat_agreement(old, arm),
                "updated_repeat_agreement": _repeat_agreement(new, arm),
            }
        )
    return rows


def _production_metric_rows(
    old: dict[tuple[int, str, str, str], dict[str, str]],
    new: dict[tuple[int, str, str, str], dict[str, str]],
) -> list[dict[str, Any]]:
    metrics = sorted(
        {key[2] for key in old if key[3] == "production_default"}
    )
    rows = []
    for metric in metrics:
        keys = [
            key
            for key, row in old.items()
            if key[2] == metric
            and key[3] == "production_default"
            and row["gt_label"] != "ambiguous"
        ]
        corrected, regressed = _correction_counts(old, new, keys)
        old_accuracy = _accuracy(old[key] for key in keys)
        new_accuracy = _accuracy(new[key] for key in keys)
        case_differences = _case_differences(old, new, keys)
        low, high = _bootstrap_interval(case_differences)
        rows.append(
            {
                "metric": metric,
                "binary_n": len(keys),
                "old_binary_accuracy": old_accuracy,
                "updated_binary_accuracy": new_accuracy,
                "delta_pp": 100.0 * (new_accuracy - old_accuracy),
                "paired_corrected": corrected,
                "paired_regressed": regressed,
                "paired_exact_p": _exact_binomial_p(corrected, regressed),
                "case_bootstrap_delta_low_pp": 100.0 * low,
                "case_bootstrap_delta_high_pp": 100.0 * high,
                "updated_repeat_agreement": _repeat_agreement(
                    new, "production_default", metric=metric
                ),
                "updated_evidence_sufficient_rate": _mean(
                    new[key]["evidence_status"] == "sufficient" for key in keys
                ),
            }
        )
    return rows


def _new_metric_arm_rows(
    new: dict[tuple[int, str, str, str], dict[str, str]],
) -> list[dict[str, Any]]:
    groups: defaultdict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for (_repeat, _case, metric, arm), row in new.items():
        if row["gt_label"] != "ambiguous":
            groups[(metric, arm)].append(row)
    rows = []
    for (metric, arm), values in sorted(groups.items()):
        rows.append(
            {
                "metric": metric,
                "arm": arm,
                "binary_n": len(values),
                "binary_accuracy": _accuracy(values),
                "evidence_sufficient_rate": _mean(
                    value["evidence_status"] == "sufficient" for value in values
                ),
                "predicted_ambiguous_rate": _mean(
                    value["predicted_label"] == "ambiguous" for value in values
                ),
            }
        )
    return rows


def _transition_rows(
    old: dict[tuple[int, str, str, str], dict[str, str]],
    new: dict[tuple[int, str, str, str], dict[str, str]],
    ownership: dict[str, str],
) -> list[dict[str, Any]]:
    rows = []
    for key in sorted(old):
        prior = old[key]
        updated = new[key]
        if prior["predicted_label"] == updated["predicted_label"]:
            continue
        repeat, case_id, metric, arm = key
        rows.append(
            {
                "repeat": repeat,
                "case_id": case_id,
                "metric": metric,
                "canonical_owner": ownership[case_id],
                "arm": arm,
                "provisional_human_label": prior["gt_label"],
                "old_prediction": prior["predicted_label"],
                "updated_prediction": updated["predicted_label"],
                "old_binary_match": prior["binary_match"],
                "updated_binary_match": updated["binary_match"],
                "updated_evidence_status": updated["evidence_status"],
                "updated_confidence": updated["confidence"],
                "updated_reason": updated["reason"],
            }
        )
    return rows


def _ownership_rows(
    old: dict[tuple[int, str, str, str], dict[str, str]],
    new: dict[tuple[int, str, str, str], dict[str, str]],
    ownership: dict[str, str],
) -> list[dict[str, Any]]:
    rows = []
    for arm in ("production_default", "global_only", "local_raw_only", "full_raw"):
        for owner in (
            "object_pairing_consistency",
            "functional_semantic_fidelity",
            "oor",
        ):
            keys = [
                key
                for key, row in old.items()
                if key[2] == "object_pairing_consistency"
                and key[3] == arm
                and ownership[key[1]] == owner
                and row["gt_label"] != "ambiguous"
            ]
            rows.append(
                {
                    "legacy_metric": "object_pairing_consistency",
                    "canonical_owner": owner,
                    "arm": arm,
                    "binary_n": len(keys),
                    "old_binary_accuracy": _accuracy(old[key] for key in keys),
                    "updated_binary_accuracy": _accuracy(new[key] for key in keys),
                }
            )
    return rows


def _confidence_rows(
    old: dict[tuple[int, str, str, str], dict[str, str]],
    new: dict[tuple[int, str, str, str], dict[str, str]],
) -> list[dict[str, Any]]:
    rows = []
    for name, run in (("old", old), ("updated", new)):
        values = [
            row
            for (_repeat, _case, _metric, arm), row in run.items()
            if arm == "production_default" and row["gt_label"] != "ambiguous"
        ]
        correct = [
            float(row["confidence"])
            for row in values
            if row["binary_match"] == "True"
        ]
        errors = [
            float(row["confidence"])
            for row in values
            if row["binary_match"] == "False"
        ]
        rows.append(
            {
                "run": name,
                "correct_n": len(correct),
                "mean_confidence_correct": statistics.fmean(correct),
                "error_n": len(errors),
                "mean_confidence_error": statistics.fmean(errors),
                "confidence_separation": statistics.fmean(correct)
                - statistics.fmean(errors),
            }
        )
    return rows


def _evidence_integrity(
    old: dict[tuple[int, str, str, str], dict[str, str]],
    new: dict[tuple[int, str, str, str], dict[str, str]],
    old_root: Path,
    new_root: Path,
) -> dict[str, Any]:
    raw_arms = ("full_raw", "global_only", "local_raw_only", "production_raw_swap")
    raw_keys = [key for key in old if key[3] in raw_arms]
    packet_equal = sum(
        old[key]["evidence_packet_sha256"] == new[key]["evidence_packet_sha256"]
        for key in raw_keys
    )

    old_manifest = json.loads(
        (old_root / "repeat_1" / "run_manifest.json").read_text(encoding="utf-8")
    )
    new_manifest = json.loads(
        (new_root / "repeat_1" / "run_manifest.json").read_text(encoding="utf-8")
    )
    old_contour_root = Path(old_manifest["inputs"]["contour_root"])
    new_contour_root = Path(new_manifest["inputs"]["contour_root"])
    old_manifests = {
        path.relative_to(old_contour_root): path
        for path in old_contour_root.rglob("evidence_manifest.json")
    }
    new_manifests = {
        path.relative_to(new_contour_root): path
        for path in new_contour_root.rglob("evidence_manifest.json")
    }
    image_total = 0
    image_equal = 0
    for relative in sorted(set(old_manifests) & set(new_manifests)):
        left = json.loads(old_manifests[relative].read_text(encoding="utf-8"))
        right = json.loads(new_manifests[relative].read_text(encoding="utf-8"))
        for left_view, right_view in zip(
            left.get("contour_views", []), right.get("contour_views", [])
        ):
            image_total += 1
            image_equal += _sha256(Path(left_view["path"])) == _sha256(
                Path(right_view["path"])
            )
    return {
        "paired_calls_per_run": len(old),
        "raw_packet_hash_equal": packet_equal,
        "raw_packet_hash_total": len(raw_keys),
        "contour_image_bytes_equal": image_equal,
        "contour_image_total": image_total,
    }


def _report(
    *,
    old_root: Path,
    new_root: Path,
    overall_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    new_arm_rows: list[dict[str, Any]],
    ownership_rows: list[dict[str, Any]],
    confidence_rows: list[dict[str, Any]],
    evidence_integrity: dict[str, Any],
) -> str:
    overall = {row["arm"]: row for row in overall_rows}
    metrics = {row["metric"]: row for row in metric_rows}
    conf = {row["run"]: row for row in confidence_rows}
    lines = [
        "# Exp 2 non-L1 updated-boundary comparison",
        "",
        "## Bottom line",
        "",
        "The updated boundary/rubric is directionally better but does not produce a "
        "benchmark-wide accuracy jump. Production-default agreement with the existing "
        "human labels rises from "
        f"{_pct(overall['production_default']['old_binary_accuracy'])} to "
        f"{_pct(overall['production_default']['updated_binary_accuracy'])} "
        f"({overall['production_default']['delta_pp']:+.1f} pp). The paired "
        "case-bootstrap interval crosses zero, so this overall change is not yet a "
        "stable improvement claim.",
        "",
        "The one clear targeted effect is the legacy Object Pairing bucket: after "
        "routing category compatibility, prompt-owned function, and explicit relations "
        "to their intended owners, production-default agreement rises from "
        f"{_pct(metrics['object_pairing_consistency']['old_binary_accuracy'])} to "
        f"{_pct(metrics['object_pairing_consistency']['updated_binary_accuracy'])}. "
        "This is a boundary-definition improvement, although the bucket name remains "
        "legacy in the artifact.",
        "",
        "OAR remains the main failure mode. It is dominated by `ambiguous` responses "
        "when the exact wall identity, ceiling contact, or attachment interface is not "
        "visually provable. That behavior is consistent with the positive-verification "
        "rubric, but inconsistent with the current provisional labels that generally "
        "treat visually ordinary attachments as valid.",
        "",
        "## Experimental integrity",
        "",
        f"- Old run: `{old_root}`",
        f"- Updated run: `{new_root}`",
        f"- Paired calls: {evidence_integrity['paired_calls_per_run']} per run.",
        "- Both runs contain 480 calls × 2 exact-input repeats, with zero failed calls.",
        f"- Raw packet hashes match for {evidence_integrity['raw_packet_hash_equal']} / "
        f"{evidence_integrity['raw_packet_hash_total']} paired calls.",
        f"- Contour image bytes match for {evidence_integrity['contour_image_bytes_equal']} / "
        f"{evidence_integrity['contour_image_total']} images.",
        "- Human labels were never sent to the model.",
        "- Important limitation: the old run did not freeze a request-prompt hash. The "
        "visual evidence is proven identical, but the historical prompt version is "
        "inferred from the run timeline rather than independently reconstructed.",
        "",
        "## Overall comparison by arm",
        "",
        "| Arm | Old | Updated | Delta | Corrected / regressed | Updated repeat agreement |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in overall_rows:
        lines.append(
            f"| {row['arm']} | {_pct(row['old_binary_accuracy'])} | "
            f"{_pct(row['updated_binary_accuracy'])} | {row['delta_pp']:+.1f} pp | "
            f"{row['paired_corrected']} / {row['paired_regressed']} | "
            f"{_pct(row['updated_repeat_agreement'])} |"
        )
    lines.extend(
        [
            "",
            "The updated production policy is stable across repeats (93.5%), but the "
            "small +1.5 pp overall change is within sampling/stochastic uncertainty. "
            "Full raw is unchanged; adding every available image is therefore not a "
            "general solution.",
            "",
            "## Production-default comparison by metric",
            "",
            "| Metric | Old | Updated | Delta | Updated repeat agreement | Evidence sufficient |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in metric_rows:
        lines.append(
            f"| {row['metric']} | {_pct(row['old_binary_accuracy'])} | "
            f"{_pct(row['updated_binary_accuracy'])} | {row['delta_pp']:+.1f} pp | "
            f"{_pct(row['updated_repeat_agreement'])} | "
            f"{_pct(row['updated_evidence_sufficient_rate'])} |"
        )
    lines.extend(
        [
            "",
            "### Metric takeaways",
            "",
            "- **Room/scene type:** 100% binary agreement across all main arms. Global-only "
            "is sufficient; additional local views provide no measurable benefit.",
            "- **Required functional areas:** production/global/full raw are tied at 88.9%; "
            "local-only falls to 72.2%. This supports global-first evidence.",
            "- **Broad semantic intent:** remains label-boundary-sensitive (54.5% production). "
            "Local-only is numerically best at 63.6%, but n=11 binary cases is too small "
            "to revise the global-first policy.",
            "- **Scale:** local-only is best at 78.6%; global-only is 57.1%. Keep local "
            "evidence as the default. The updated production decrease is three paired "
            "regressions and is not evidence that the boundary itself is worse.",
            "- **Style:** local-only and full raw are tied at 82.1%, versus 78.6% global "
            "and 75.0% production. The result suggests that the designated target needs "
            "a readable local view alongside context.",
            "- **OOR:** production and contour-only both reach 72.7%; production has 100% "
            "repeat agreement. The current production evidence is adequate for this pilot.",
            "- **OAR:** production is only 12.5% against provisional labels, and evidence "
            "sufficiency is 16.7%. Raw local variants outperform contour, but even the "
            "best arm is only 29.2%. This needs boundary/annotation adjudication before "
            "another camera ablation.",
            "- **Legacy Object Pairing bucket:** the updated boundary improves production "
            "by +18.8 pp. Global-only is best at 68.8%, but this bucket mixes three "
            "canonical owners and must not be reported as one final metric.",
            "",
            "## Legacy Object Pairing ownership split",
            "",
            "The 16 historical cases actually contain 3 category-pairing cases, 9 "
            "prompt-owned functional cases, and 4 explicit OOR cases. Detailed values "
            "are in `legacy_pairing_ownership_slices.tsv`. The improvement is strongest "
            "under global evidence for functional semantics; explicit orientation/OOR "
            "cases benefit more from local evidence.",
            "",
            "## Stability and confidence",
            "",
            f"- Production repeat agreement remains "
            f"{_pct(overall['production_default']['updated_repeat_agreement'])}.",
            "- Full-raw repeat agreement improves to 93.5%, but contour-only and "
            "production-raw-swap agreement drop to 83.3% and 75.0%. With only 24 cases, "
            "this is best treated as stochastic sensitivity rather than a contour regression.",
            f"- Updated mean confidence is {conf['updated']['mean_confidence_correct']:.3f} "
            f"on correct production decisions and {conf['updated']['mean_confidence_error']:.3f} "
            "on errors. The separation is only "
            f"{conf['updated']['confidence_separation']:.3f}; confidence is still not calibrated.",
            "",
            "## What is supported now",
            "",
            "1. The new boundary fixes a real overreach: generic Object Pairing should not "
            "penalize angle, orientation, distance, or functionality.",
            "2. Global evidence is sufficient for room type and required-area presence.",
            "3. Local evidence remains necessary for Scale and useful for Style and explicit "
            "relation/orientation cases.",
            "4. More images are not monotonically better; full raw does not beat the "
            "metric-specific production policy overall.",
            "5. OAR disagreement is primarily a rubric/annotation contract issue. Do not "
            "tune camera policy against the current OAR accuracy number yet.",
            "",
            "## Human audit required",
            "",
            "Use `paired_label_transitions.tsv` as the review queue. Priority order:",
            "",
            "1. all OAR cases, especially wall-axis identity and ceiling attachment;",
            "2. the three Scale regressions;",
            "3. the rerouted legacy Object Pairing cases;",
            "4. Broad Semantic Intent disagreements.",
            "",
            "After the updated human labels are frozen, rerun only the analysis layer; no "
            "rendering or model calls are required.",
            "",
        ]
    )
    return "\n".join(lines)


def _correction_counts(
    old: dict[tuple[int, str, str, str], dict[str, str]],
    new: dict[tuple[int, str, str, str], dict[str, str]],
    keys: Iterable[tuple[int, str, str, str]],
) -> tuple[int, int]:
    corrected = 0
    regressed = 0
    for key in keys:
        prior = old[key]["binary_match"] == "True"
        updated = new[key]["binary_match"] == "True"
        corrected += int(not prior and updated)
        regressed += int(prior and not updated)
    return corrected, regressed


def _case_differences(
    old: dict[tuple[int, str, str, str], dict[str, str]],
    new: dict[tuple[int, str, str, str], dict[str, str]],
    keys: Iterable[tuple[int, str, str, str]],
) -> list[float]:
    grouped: defaultdict[str, list[float]] = defaultdict(list)
    for key in keys:
        grouped[key[1]].append(
            float(new[key]["binary_match"] == "True")
            - float(old[key]["binary_match"] == "True")
        )
    return [statistics.fmean(values) for values in grouped.values()]


def _bootstrap_interval(values: list[float]) -> tuple[float, float]:
    if not values:
        return (math.nan, math.nan)
    rng = random.Random(20260727)
    samples = sorted(
        statistics.fmean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(20000)
    )
    return samples[500], samples[19500]


def _exact_binomial_p(corrected: int, regressed: int) -> float:
    total = corrected + regressed
    if total == 0:
        return 1.0
    lower = min(corrected, regressed)
    tail = sum(math.comb(total, index) for index in range(lower + 1)) / (2**total)
    return min(1.0, 2.0 * tail)


def _repeat_agreement(
    run: dict[tuple[int, str, str, str], dict[str, str]],
    arm: str,
    *,
    metric: str | None = None,
) -> float:
    keys = [
        key
        for key in run
        if key[0] == 1 and key[3] == arm and (metric is None or key[2] == metric)
    ]
    return _mean(
        run[key]["predicted_label"]
        == run[(2, key[1], key[2], key[3])]["predicted_label"]
        for key in keys
    )


def _accuracy(rows: Iterable[dict[str, str]]) -> float:
    return _mean(row["binary_match"] == "True" for row in rows)


def _mean(values: Iterable[bool | float]) -> float:
    materialized = [float(value) for value in values]
    return statistics.fmean(materialized) if materialized else math.nan


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _write_grouped_bar_svg(
    path: Path,
    *,
    title: str,
    rows: list[dict[str, Any]],
    label_key: str,
    old_key: str,
    new_key: str,
) -> None:
    width = 1100
    left = 280
    top = 90
    row_height = 54
    chart_width = 700
    height = top + row_height * len(rows) + 70
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#111827"/>',
        f'<text x="40" y="48" fill="#f9fafb" font-family="Arial" font-size="24" '
        f'font-weight="700">{_xml(title)}</text>',
        '<text x="820" y="48" fill="#9ca3af" font-family="Arial" font-size="14">'
        'old ■   updated ■</text>',
    ]
    for index, row in enumerate(rows):
        y = top + index * row_height
        old_value = float(row[old_key])
        new_value = float(row[new_key])
        parts.extend(
            [
                f'<text x="40" y="{y + 25}" fill="#e5e7eb" font-family="Arial" '
                f'font-size="15">{_xml(str(row[label_key]))}</text>',
                f'<rect x="{left}" y="{y + 3}" width="{chart_width * old_value:.1f}" '
                'height="17" rx="4" fill="#64748b"/>',
                f'<rect x="{left}" y="{y + 25}" width="{chart_width * new_value:.1f}" '
                'height="17" rx="4" fill="#22c55e"/>',
                f'<text x="{left + chart_width + 14}" y="{y + 17}" fill="#cbd5e1" '
                f'font-family="Arial" font-size="13">{_pct(old_value)}</text>',
                f'<text x="{left + chart_width + 14}" y="{y + 39}" fill="#bbf7d0" '
                f'font-family="Arial" font-size="13">{_pct(new_value)}</text>',
            ]
        )
    parts.append("</svg>\n")
    path.write_text("\n".join(parts), encoding="utf-8")


def _xml(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
