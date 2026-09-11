"""Resume eligibility checks for camera-cal case runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def resumable_case(
    manifest: dict[str, Any],
    *,
    expected_fingerprint: str,
    case_out: Path,
) -> bool:
    return bool(
        manifest.get("status") == "complete"
        and manifest.get("input_fingerprint") == expected_fingerprint
        and (case_out / "evaluation_report.json").is_file()
        and (case_out / "grouping.json").is_file()
        and (case_out / "l1_report.json").is_file()
        and (case_out / "l1_diagnostics.json").is_file()
        and (case_out / "scene_quality_report.json").is_file()
        and (case_out / "scene_comparison.json").is_file()
        and (case_out / "control_manifest.json").is_file()
    )


__all__ = ["resumable_case"]
