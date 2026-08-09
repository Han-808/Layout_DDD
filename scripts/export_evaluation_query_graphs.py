#!/usr/bin/env python3
"""Export optional post-hoc audit graphs from a completed scene-level run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from benchmark.visual_judge.graphs import (  # noqa: E402
    export_case_audit_graphs,
)


def main() -> None:
    args = parse_args()
    run_root = args.run_root.expanduser().resolve()
    cases_root = run_root / "cases"
    selected = set(args.case_id)
    results: list[dict[str, Any]] = []
    for case_dir in sorted(path for path in cases_root.iterdir() if path.is_dir()):
        if selected and case_dir.name not in selected:
            continue
        result = export_case_audit_graphs(
            case_id=case_dir.name,
            grouping_report=_read_json(case_dir / "grouping.json"),
            scene_quality_report=_read_json(
                case_dir / "scene_quality_report.json"
            ),
            output_dir=case_dir / "audit_graphs",
        )
        results.append(result)
        print(
            json.dumps(
                {
                    "case_id": case_dir.name,
                    "status": result["status"],
                    "query_graph_count": len(
                        result["evaluation_query_graphs"]
                    ),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    if selected - {str(item["case_id"]) for item in results}:
        missing = sorted(
            selected - {str(item["case_id"]) for item in results}
        )
        raise SystemExit(f"case directories not found: {', '.join(missing)}")
    if any(item["status"] != "complete" for item in results):
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Repeat to export selected cases; omit to export every case.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


if __name__ == "__main__":
    main()
