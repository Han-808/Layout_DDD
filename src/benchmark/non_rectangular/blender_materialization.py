"""Blender subprocess adapter for non-rectangular room materialization."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any

from benchmark.materialization.catalog import sha256_file
from benchmark.non_rectangular.materialization import (
    NonRectangularMaterializationContractError,
    NonRectangularMaterializationInfrastructureError,
)


class BlenderNonRectangularRoomMaterializer:
    """Build and independently inspect one room-scoped sanitized blend."""

    def materialize(
        self,
        *,
        plan_path: Path,
        blend_path: Path,
        inspection_path: Path,
        blender_bin: Path,
        timeout_seconds: int,
    ) -> dict[str, Any]:
        plan = plan_path.expanduser().resolve()
        blend = blend_path.expanduser().resolve()
        inspection = inspection_path.expanduser().resolve()
        executable = blender_bin.expanduser().resolve()
        if not plan.is_file() or not executable.is_file():
            raise NonRectangularMaterializationContractError(
                "materialization plan or Blender executable is missing"
            )
        if len({plan, blend, inspection}) != 3:
            raise NonRectangularMaterializationContractError(
                "plan, blend, and inspection paths must be distinct"
            )
        blend.parent.mkdir(parents=True, exist_ok=True)
        inspection.parent.mkdir(parents=True, exist_ok=True)
        plan_hash_before = sha256_file(plan)
        build_report_path = blend.with_suffix(blend.suffix + ".build.json")
        build_worker = Path(__file__).with_name(
            "nonrect_catalog_materializer_worker.py"
        )
        build = [
            str(executable),
            "--background",
            "--factory-startup",
            "--disable-autoexec",
            "--python-exit-code",
            "1",
            "--python",
            str(build_worker),
            "--",
            "--plan-json",
            str(plan),
            "--out-blend",
            str(blend),
            "--report-json",
            str(build_report_path),
        ]
        _run(
            build,
            timeout_seconds=timeout_seconds,
            stdout_path=blend.with_suffix(blend.suffix + ".build.stdout.log"),
            stderr_path=blend.with_suffix(blend.suffix + ".build.stderr.log"),
            stage="materialization",
        )
        if sha256_file(plan) != plan_hash_before:
            raise NonRectangularMaterializationContractError(
                "Blender materializer modified the read-only plan"
            )
        if not blend.is_file() or not build_report_path.is_file():
            raise NonRectangularMaterializationInfrastructureError(
                "filesystem_interruption",
                "Blender materializer omitted blend or build report",
            )
        build_report = _read_json(build_report_path)
        if (
            build_report.get("status") != "built"
            or build_report.get("render_invocation_count") != 0
        ):
            raise NonRectangularMaterializationContractError(
                "Blender build report does not prove build-only execution"
            )

        inspect_worker = Path(__file__).with_name(
            "nonrect_blend_inspector_worker.py"
        )
        inspect = [
            str(executable),
            "--background",
            "--disable-autoexec",
            str(blend),
            "--python-exit-code",
            "1",
            "--python",
            str(inspect_worker),
            "--",
            "--plan-json",
            str(plan),
            "--out-json",
            str(inspection),
        ]
        _run(
            inspect,
            timeout_seconds=timeout_seconds,
            stdout_path=blend.with_suffix(blend.suffix + ".inspect.stdout.log"),
            stderr_path=blend.with_suffix(blend.suffix + ".inspect.stderr.log"),
            stage="inspection",
        )
        report = _read_json(inspection)
        if report.get("status") != "passed":
            raise NonRectangularMaterializationContractError(
                "non-rectangular independent blend inspection failed"
            )
        report["materialization"] = {
            "status": "built_and_independently_inspected",
            "plan_sha256_before": plan_hash_before,
            "plan_sha256_after": sha256_file(plan),
            "trusted_blend_sha256": sha256_file(blend),
            "build_report_sha256": sha256_file(build_report_path),
        }
        inspection.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return report


def _run(
    command: list[str],
    *,
    timeout_seconds: int,
    stdout_path: Path,
    stderr_path: Path,
    stage: str,
) -> None:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise NonRectangularMaterializationInfrastructureError(
            "blender_timeout",
            f"Blender {stage} timed out",
        ) from exc
    except OSError as exc:
        raise NonRectangularMaterializationInfrastructureError(
            "renderer_startup",
            f"Blender {stage} could not start",
        ) from exc
    stdout_path.write_bytes(completed.stdout or b"")
    stderr_path.write_bytes(completed.stderr or b"")
    if completed.returncode != 0:
        raise NonRectangularMaterializationInfrastructureError(
            "blender_process_crash",
            f"Blender {stage} exited with code {completed.returncode}",
        )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise NonRectangularMaterializationInfrastructureError(
            "filesystem_interruption",
            f"cannot read Blender stage report {path.name}",
        ) from exc
    if not isinstance(value, dict):
        raise NonRectangularMaterializationContractError(
            f"Blender stage report must be a JSON object: {path.name}"
        )
    return value


__all__ = ["BlenderNonRectangularRoomMaterializer"]
