#!/usr/bin/env python3
"""Build event GT for a controlled scene distortion from its frozen transform."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--distortion-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    report = _read_json(args.source_report)
    distortion = _read_json(args.distortion_manifest)
    expected = distortion.get("expected", {})
    invalid_keys = {
        "collision": {
            "|".join(sorted(str(value) for value in pair))
            for pair in expected.get("collision_invalid_pairs", [])
        },
        "oob": {str(value) for value in expected.get("oob_invalid_object_ids", [])},
        "support": {str(value) for value in expected.get("support_invalid_object_ids", [])},
    }
    metrics = report.get("reports", {}).get("generic_validity", {}).get("metrics", {})
    sources = {
        "collision": metrics.get("collision", {}).get("pairs", []),
        "oob": metrics.get("oob", {}).get("objects", []),
        "support": metrics.get("support", {}).get("objects", []),
    }
    events: list[dict[str, Any]] = []
    observed: dict[str, set[str]] = {name: set() for name in sources}

    for metric, items in sources.items():
        for item in items if isinstance(items, list) else []:
            if not isinstance(item, dict):
                continue
            judge_result = item.get("judge_result")
            request = judge_result.get("request") if isinstance(judge_result, dict) else None
            if not isinstance(request, dict):
                continue
            if metric == "collision":
                object_ids = sorted([str(item.get("object_a")), str(item.get("object_b"))])
                event_id = "|".join(object_ids)
            else:
                object_ids = [str(item.get("object_id"))]
                event_id = object_ids[0]
            observed[metric].add(event_id)
            events.append(
                {
                    "metric": metric,
                    "event_id": event_id,
                    "object_ids": object_ids,
                    "label": "invalid" if event_id in invalid_keys[metric] else "valid",
                    "reason_code": "controlled_transform_gt",
                    "review_notes": (
                        "Invalid label comes from the frozen metric-specific transform."
                        if event_id in invalid_keys[metric]
                        else "Routed non-target event remains valid under the frozen source-scene contract."
                    ),
                    "frozen_request": request,
                }
            )

    missed = {
        metric: sorted(invalid_keys[metric] - observed[metric])
        for metric in invalid_keys
        if invalid_keys[metric] - observed[metric]
    }
    if missed:
        raise RuntimeError(
            "controlled invalid events were not routed by the source evaluation: "
            + json.dumps(missed, sort_keys=True)
        )

    payload = {
        "schema_version": "p0b_camera_ablation_gt_v1",
        "status": "confirmed_programmatic",
        "case_id": distortion["case_id"],
        "source_report": str(args.source_report.resolve()),
        "distortion_manifest": str(args.distortion_manifest.resolve()),
        "ground_truth_source": "frozen_controlled_transform",
        "review_contract": {
            "allowed_labels": ["valid", "invalid"],
            "positive_class": "invalid",
            "human_review_required": False,
        },
        "events": events,
        "coverage": {
            "expected_invalid": {key: sorted(value) for key, value in invalid_keys.items()},
            "observed_routed": {key: sorted(value) for key, value in observed.items()},
            "missed_expected": {},
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"GT: {args.out} events={len(events)} invalid={sum(e['label'] == 'invalid' for e in events)}")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


if __name__ == "__main__":
    main()
