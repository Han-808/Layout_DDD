"""Validate a frozen reference annotation without modifying it.

Summary:
    Read-only check that a reference annotation is well-formed and reports its
    scoring gate (whether it is officially scoreable).

Input:
    - ``--annotation``: reference_annotation JSON.

Output:
    - Prints the scoring-gate result JSON to stdout. No files written.

Function:
    Thin CLI wrapper over the reference-annotation validation/gate helpers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from benchmark.reference_annotation import (  # noqa: E402
    annotation_scoring_gate,
    validate_reference_annotation,
)
from benchmark.utils.io import read_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a frozen benchmark reference annotation without modifying it."
    )
    parser.add_argument("--annotation", required=True)
    parser.add_argument(
        "--require-scoreable",
        action="store_true",
        help="Exit nonzero unless the annotation is confirmed and eligible for official scoring.",
    )
    args = parser.parse_args()

    annotation = read_json(_path(args.annotation))
    validate_reference_annotation(annotation)
    gate = annotation_scoring_gate(annotation)
    print(json.dumps(gate, indent=2, sort_keys=True))
    if args.require_scoreable and not gate["official_scoreable"]:
        raise SystemExit(2)


def _path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path.cwd() / path


if __name__ == "__main__":
    main()
