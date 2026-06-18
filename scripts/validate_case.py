from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from jsonschema import Draft202012Validator

from benchmark.utils.io import load_json_schema, read_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a benchmark case JSON file.")
    parser.add_argument("--case", required=True)
    args = parser.parse_args()

    case = read_json(args.case)
    schema = load_json_schema(PROJECT_ROOT / "schemas" / "bm_instance.schema.json")
    errors = sorted(Draft202012Validator(schema).iter_errors(case), key=lambda e: list(e.path))
    if errors:
        for error in errors:
            path = "$" + "".join(f"[{p!r}]" if isinstance(p, int) else f".{p}" for p in error.path)
            print(f"{path}: {error.message}")
        raise SystemExit(1)
    print(f"valid: {args.case}")


if __name__ == "__main__":
    main()
