"""Approve and freeze a drafted reference annotation for official scoring.

Summary:
    Human-in-the-loop step that turns a reviewed annotation draft into a frozen,
    scoreable reference annotation.

Input:
    - ``--draft`` annotation JSON, ``--reviewer``, and optional ``--reviewed-at``.

Output:
    - ``--out``: approved reference_annotation JSON; prints the scoring gate.

Function:
    Records reviewer/confirmation metadata, validates the annotation, and writes
    the frozen artifact that the evaluator will treat as ground truth.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from benchmark.reference_annotation import (  # noqa: E402
    INVENTORY_POLICIES,
    annotation_scoring_gate,
    approve_reference_annotation,
)
from benchmark.utils.io import read_json, write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze an already human-reviewed reference-annotation draft. "
            "This command preserves every claim_state and performs no inference."
        )
    )
    parser.add_argument("--draft", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--reviewed-at", default=None)
    parser.add_argument(
        "--inventory-policy",
        choices=INVENTORY_POLICIES,
        required=True,
        help=(
            "closed_world scores missing and extra object inventory claims; "
            "open_world scores only explicitly confirmed claims."
        ),
    )
    args = parser.parse_args()

    approved = approve_reference_annotation(
        read_json(_path(args.draft)),
        inventory_policy=args.inventory_policy,
        reviewer=args.reviewer,
        reviewed_at=args.reviewed_at,
    )
    output = write_json(_path(args.out), approved)
    gate = annotation_scoring_gate(approved)
    print(f"reference_annotation: {output}")
    print(json.dumps(gate, indent=2, sort_keys=True))


def _path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path.cwd() / path


if __name__ == "__main__":
    main()
