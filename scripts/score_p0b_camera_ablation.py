#!/usr/bin/env python3
"""Score P0b camera-policy replay outputs against frozen event GT."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--metric",
        action="append",
        choices=("collision", "oob", "support"),
        default=[],
        help="Restrict scoring to selected frozen metrics.",
    )
    args = parser.parse_args()
    gt = _read_json(args.gt)
    run_dir = Path(args.run_dir).expanduser().resolve()
    selected_metrics = set(args.metric)
    gt_events = {
        (str(item["metric"]), str(item["event_id"])): item
        for item in gt.get("events") or []
        if not selected_metrics or str(item.get("metric")) in selected_metrics
    }
    if not gt_events:
        raise ValueError("GT fixture contains no events")

    per_event = []
    summaries = []
    for manifest_path in sorted(run_dir.glob("*/mode_results.json")):
        manifest = _read_json(manifest_path)
        mode = str(manifest.get("arm") or manifest.get("mode") or manifest_path.parent.name)
        camera_mode = str(manifest.get("camera_mode") or manifest.get("mode") or "")
        evidence_style = str(manifest.get("evidence_style") or "legacy")
        results = {
            (str(item.get("metric")), str(item.get("event_id"))): item
            for item in manifest.get("results") or []
            if isinstance(item, dict)
        }
        rows = []
        for key, gt_event in gt_events.items():
            result = results.get(key, {})
            predicted = result.get("predicted_label")
            label = gt_event["label"]
            resolved = predicted in {"valid", "invalid"}
            match = resolved and predicted == label
            row = {
                "mode": mode,
                "arm": mode,
                "camera_mode": camera_mode,
                "resolved_camera_mode": result.get("resolved_camera_mode", camera_mode),
                "evidence_style": evidence_style,
                "local_presentation": result.get("local_presentation", ""),
                "global_context": result.get("global_context", ""),
                "image_order": result.get("image_order", ""),
                "final_image_budget": result.get("final_image_budget", ""),
                "max_local_views": result.get("max_local_views", ""),
                "metric": key[0],
                "event_id": key[1],
                "gt_label": label,
                "predicted_label": predicted or "missing",
                "match": int(match),
                "resolved": int(resolved),
                "confidence": result.get("confidence", ""),
                "image_count": result.get("image_count", ""),
                "camera_evidence_seconds": result.get("camera_evidence_seconds", ""),
                "candidate_preview_seconds": result.get("candidate_preview_seconds", ""),
                "selector_seconds": result.get("selector_seconds", ""),
                "final_render_seconds": result.get("final_render_seconds", ""),
                "judge_seconds": result.get("judge_seconds", ""),
                "elapsed_seconds": result.get("elapsed_seconds", ""),
                "estimated_uncached_seconds": result.get("estimated_uncached_seconds", ""),
                "camera_max_steps": result.get("camera_max_steps", ""),
                "pose_selector_enabled": result.get("pose_selector_enabled", ""),
                "frozen_event_packet_sha256": result.get("frozen_event_packet_sha256", ""),
                "frozen_scene_sha256": result.get("frozen_scene_sha256", ""),
                "frozen_source_report_sha256": result.get("frozen_source_report_sha256", ""),
                "frozen_gt_sha256": result.get("frozen_gt_sha256", ""),
                "final_judge_model": _compact_json(result.get("final_judge_model")),
                "pose_selector_model": _compact_json(result.get("pose_selector_model")),
                "error": result.get("error", ""),
                "gt_reason_code": gt_event.get("reason_code", ""),
            }
            rows.append(row)
            per_event.append(row)

        evidence_dir = Path(str(manifest.get("camera_evidence_dir") or manifest_path.parent / "camera_evidence"))
        fallback_count, degraded_count = _camera_degradation_counts(evidence_dir)
        for metric in ["overall", "collision", "oob", "support"]:
            selected = rows if metric == "overall" else [row for row in rows if row["metric"] == metric]
            summaries.append(
                _summary_row(
                    mode,
                    metric,
                    selected,
                    elapsed_seconds=manifest.get("elapsed_seconds", "") if metric == "overall" else "",
                    fallback_events=fallback_count if metric == "overall" else "",
                    degraded_events=degraded_count if metric == "overall" else "",
                    camera_mode=camera_mode,
                    evidence_style=evidence_style,
                )
            )

    if not summaries:
        raise FileNotFoundError(f"no */mode_results.json files found under {run_dir}")
    _write_tsv(run_dir / "per_event.tsv", per_event)
    _write_tsv(run_dir / "summary.tsv", summaries)
    (run_dir / "summary.json").write_text(
        json.dumps({"per_event": per_event, "summary": summaries}, indent=2),
        encoding="utf-8",
    )
    for row in summaries:
        if row["metric"] == "overall":
            print(
                f"{row['mode']}: accuracy={row['accuracy_all']} "
                f"({row['correct']}/{row['total']}), coverage={row['coverage']}, "
                f"FP={row['fp']} FN={row['fn']}, fallbacks={row['fallback_events']}"
            )


def _summary_row(
    mode: str,
    metric: str,
    rows: list[dict[str, Any]],
    *,
    elapsed_seconds: Any,
    fallback_events: Any,
    degraded_events: Any,
    camera_mode: str = "",
    evidence_style: str = "legacy",
) -> dict[str, Any]:
    total = len(rows)
    resolved = sum(int(row["resolved"]) for row in rows)
    correct = sum(int(row["match"]) for row in rows)
    tp = sum(row["gt_label"] == "invalid" and row["predicted_label"] == "invalid" for row in rows)
    fp = sum(row["gt_label"] == "valid" and row["predicted_label"] == "invalid" for row in rows)
    fn = sum(row["gt_label"] == "invalid" and row["predicted_label"] == "valid" for row in rows)
    tn = sum(row["gt_label"] == "valid" and row["predicted_label"] == "valid" for row in rows)
    image_count = _numeric_sum(rows, "image_count")
    camera_seconds = _numeric_sum(rows, "camera_evidence_seconds")
    judge_seconds = _numeric_sum(rows, "judge_seconds")
    measured_seconds = _numeric_sum(rows, "elapsed_seconds")
    estimated_seconds = _numeric_sum(rows, "estimated_uncached_seconds")
    return {
        "mode": mode,
        "arm": mode,
        "camera_mode": camera_mode,
        "evidence_style": evidence_style,
        "metric": metric,
        "total": total,
        "resolved": resolved,
        "correct": correct,
        "accuracy_all": correct / total if total else 0.0,
        "accuracy_resolved": correct / resolved if resolved else 0.0,
        "coverage": resolved / total if total else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "elapsed_seconds": elapsed_seconds,
        "measured_event_seconds": measured_seconds,
        "camera_evidence_seconds": camera_seconds,
        "judge_seconds": judge_seconds,
        "estimated_uncached_seconds": estimated_seconds,
        "image_count": image_count,
        "mean_estimated_uncached_seconds": estimated_seconds / total if total else 0.0,
        "mean_images_per_event": image_count / total if total else 0.0,
        "fallback_events": fallback_events,
        "degraded_events": degraded_events,
        "error_events": sum(bool(str(row.get("error") or "").strip()) for row in rows),
    }


def _camera_degradation_counts(evidence_dir: Path) -> tuple[int, int]:
    fallback_count = 0
    degraded_count = 0
    for path in evidence_dir.rglob("camera_evidence_manifest.json"):
        manifest = _read_json(path)
        selection = manifest.get("selection") if isinstance(manifest.get("selection"), dict) else {}
        fallback = selection.get("fallback_reason") or (selection.get("ranking") or {}).get("fallback_reason")
        selector = str(selection.get("selector") or "")
        if fallback or "fallback" in selector:
            fallback_count += 1
        if manifest.get("highlight_degradation_reason") or manifest.get("overlay_degradation_reason"):
            degraded_count += 1
    return fallback_count, degraded_count


def _numeric_sum(rows: list[dict[str, Any]], key: str) -> float:
    total = 0.0
    for row in rows:
        value = row.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            total += float(value)
            continue
        if isinstance(value, str) and value.strip():
            try:
                total += float(value)
            except ValueError:
                pass
    return total


def _compact_json(value: Any) -> str:
    if value is None:
        return ""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


if __name__ == "__main__":
    main()
