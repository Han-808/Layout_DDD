#!/usr/bin/env python3
"""Validate the repository's explicit pytest suite inventory.

This checker is intentionally independent from evaluator behavior.  It keeps
the current canonical pytest file list stable while making every other tracked
test visible as an explicit extended-suite member.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "configs" / "testing" / "pytest_suites_v1.json"
EXPECTED_SCHEMA_VERSION = "pytest_suites_v1"
EXPECTED_ENVIRONMENT_MARKERS = {
    "requires_blender",
    "requires_git_history",
    "requires_local_data",
    "requires_loopback",
}
_SUMMARY_LINE = re.compile(r"^(tests/[^:\s]+\.py):\s*([0-9]+)$")
_NODE_LINE = re.compile(r"^(tests/[^:\s]+\.py)::")


class InventoryError(RuntimeError):
    """Raised when the declared test inventory is incomplete or inconsistent."""


def _string_list(value: Any, *, field: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise InventoryError(f"{field} must be an array of strings")
    result = [item.strip() for item in value]
    if any(not item for item in result):
        raise InventoryError(f"{field} may not contain empty paths")
    if len(result) != len(set(result)):
        raise InventoryError(f"{field} contains duplicate paths")
    return result


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InventoryError(f"cannot read suite manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise InventoryError("suite manifest root must be an object")
    if value.get("schema_version") != EXPECTED_SCHEMA_VERSION:
        raise InventoryError(
            "suite manifest schema_version must be " f"{EXPECTED_SCHEMA_VERSION!r}"
        )
    return value


def load_pytest_config() -> dict[str, Any]:
    try:
        value = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        config = value["tool"]["pytest"]["ini_options"]
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError) as exc:
        raise InventoryError(f"cannot read tool.pytest.ini_options: {exc}") from exc
    if not isinstance(config, dict):
        raise InventoryError("tool.pytest.ini_options must be a table")
    return config


def tracked_test_paths() -> list[str]:
    completed = subprocess.run(
        [
            "git",
            "ls-files",
            "--",
            "tests/test_*.py",
            "tests/**/test_*.py",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise InventoryError(f"git ls-files failed: {detail}")
    return sorted(set(line.strip() for line in completed.stdout.splitlines() if line.strip()))


def _registered_marker_names(config: dict[str, Any]) -> set[str]:
    raw = config.get("markers", [])
    markers = _string_list(raw, field="tool.pytest.ini_options.markers")
    return {item.split(":", 1)[0].strip() for item in markers}


def validate_inventory(
    manifest: dict[str, Any],
    *,
    tracked: Iterable[str] | None = None,
) -> dict[str, list[str]]:
    canonical = _string_list(manifest.get("canonical"), field="canonical")
    extended = _string_list(manifest.get("extended"), field="extended")
    environment_markers = set(
        _string_list(manifest.get("environment_markers"), field="environment_markers")
    )

    overlap = sorted(set(canonical) & set(extended))
    if overlap:
        raise InventoryError(f"canonical and extended suites overlap: {overlap}")

    declared = set(canonical) | set(extended)
    tracked_set = set(tracked if tracked is not None else tracked_test_paths())
    missing = sorted(tracked_set - declared)
    extra = sorted(declared - tracked_set)
    if missing or extra:
        raise InventoryError(
            "suite inventory differs from tracked tests: "
            f"missing={missing}, extra={extra}"
        )

    absent_paths = sorted(path for path in declared if not (REPO_ROOT / path).is_file())
    if absent_paths:
        raise InventoryError(f"declared test files do not exist: {absent_paths}")

    pytest_config = load_pytest_config()
    configured_testpaths = _string_list(
        pytest_config.get("testpaths"), field="tool.pytest.ini_options.testpaths"
    )
    if canonical != configured_testpaths:
        raise InventoryError(
            "canonical suite must exactly preserve pyproject testpaths order"
        )

    if environment_markers != EXPECTED_ENVIRONMENT_MARKERS:
        raise InventoryError(
            "environment_markers must be exactly "
            f"{sorted(EXPECTED_ENVIRONMENT_MARKERS)}"
        )
    registered = _registered_marker_names(pytest_config)
    unregistered = sorted(environment_markers - registered)
    if unregistered:
        raise InventoryError(f"pytest markers are not registered: {unregistered}")

    return {"canonical": canonical, "extended": extended}


def collect_test_counts(paths: list[str]) -> dict[str, int]:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            *paths,
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    if completed.returncode != 0:
        raise InventoryError(f"pytest collection failed:\n{output.strip()}")

    counts: dict[str, int] = {}
    for line in output.splitlines():
        summary = _SUMMARY_LINE.match(line.strip())
        if summary:
            counts[summary.group(1)] = int(summary.group(2))
            continue
        node = _NODE_LINE.match(line.strip())
        if node:
            counts[node.group(1)] = counts.get(node.group(1), 0) + 1

    missing = sorted(set(paths) - set(counts))
    empty = sorted(path for path, count in counts.items() if path in paths and count < 1)
    if missing or empty:
        raise InventoryError(
            "tracked tests did not collect at least one node: "
            f"missing={missing}, empty={empty}"
        )
    return {path: counts[path] for path in paths}


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--skip-collect",
        action="store_true",
        help="validate paths and configuration without invoking pytest collection",
    )
    parser.add_argument(
        "--print-suite",
        choices=("canonical", "extended", "all"),
        help="print the validated suite paths, one per line",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        suites = validate_inventory(load_manifest(args.manifest))
        selected = {
            "canonical": suites["canonical"],
            "extended": suites["extended"],
            "all": sorted(suites["canonical"] + suites["extended"]),
        }
        if args.print_suite:
            print("\n".join(selected[args.print_suite]))
            return 0

        counts = {} if args.skip_collect else collect_test_counts(selected["all"])
        summary = {
            "status": "ok",
            "schema_version": EXPECTED_SCHEMA_VERSION,
            "canonical_files": len(suites["canonical"]),
            "extended_files": len(suites["extended"]),
            "tracked_files": len(selected["all"]),
            "collected_cases": sum(counts.values()) if counts else None,
            "collection_checked": not args.skip_collect,
            "environment_markers": sorted(EXPECTED_ENVIRONMENT_MARKERS),
        }
        print(json.dumps(summary, sort_keys=True))
        return 0
    except InventoryError as exc:
        print(f"pytest inventory error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
