from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from benchmark.materialization.contracts import MaterializationError


def materialize_catalog_scene(
    *,
    plan_path: Path,
    out_blend_path: Path,
    inspection_path: Path,
    blender_bin: Path,
    timeout_seconds: int = 900,
) -> dict[str, Any]:
    """Build and independently inspect one trusted fixed-catalog scene.

    The build worker starts from Blender factory state and never renders.  The
    resulting ``.blend`` is then reopened by a separate read-only worker with
    auto-execution disabled.  Full hashes are intentionally computed before and
    after each subprocess instead of relying on file metadata.
    """

    plan = _required_file(plan_path, "catalog materialization plan")
    destination = Path(out_blend_path).expanduser().resolve()
    inspection = Path(inspection_path).expanduser().resolve()
    executable = _blender_executable(blender_bin)
    _require_distinct_paths(plan, destination, inspection)
    destination.parent.mkdir(parents=True, exist_ok=True)
    inspection.parent.mkdir(parents=True, exist_ok=True)

    build_report_path = destination.with_suffix(destination.suffix + ".build.json")
    plan_hash_before = _sha256_file(plan)
    worker = Path(__file__).with_name("catalog_materializer_worker.py").resolve()
    command = [
        str(executable),
        "--background",
        "--factory-startup",
        "--disable-autoexec",
        "--python-exit-code",
        "1",
        "--python",
        str(worker),
        "--",
        "--plan-json",
        str(plan),
        "--out-blend",
        str(destination),
        "--report-json",
        str(build_report_path),
    ]
    completed = None
    worker_error: Exception | None = None
    try:
        completed = _run(
            command,
            timeout_seconds=timeout_seconds,
            stdout_path=destination.with_suffix(
                destination.suffix + ".build.stdout.log"
            ),
            stderr_path=destination.with_suffix(
                destination.suffix + ".build.stderr.log"
            ),
            label="catalog materializer",
        )
    except Exception as exc:
        worker_error = exc
    finally:
        plan_hash_after = _sha256_file(plan)
    if plan_hash_after != plan_hash_before:
        raise MaterializationError(
            "catalog materializer modified its read-only materialization plan"
        )
    if worker_error is not None:
        raise worker_error
    assert completed is not None
    if completed.returncode != 0:
        _raise_worker_error("catalog materializer", completed)
    if not destination.is_file():
        raise MaterializationError(
            f"catalog materializer did not write the trusted blend: {destination}"
        )
    build_report = _load_json_object(
        build_report_path,
        "catalog materializer build report",
    )
    if build_report.get("render_invocation_count") != 0:
        raise MaterializationError(
            "catalog materializer build report does not prove build-only execution"
        )

    blend_hash = _sha256_file(destination)
    result = inspect_sanitized_blend(
        blend_path=destination,
        expected_registry_path=plan,
        out_path=inspection,
        blender_bin=executable,
        timeout_seconds=timeout_seconds,
    )
    if result.get("status") != "passed":
        reason_codes = result.get("reason_codes")
        raise MaterializationError(
            "independent sanitized blend inspection failed"
            + (
                f": {', '.join(str(value) for value in reason_codes)}"
                if isinstance(reason_codes, list) and reason_codes
                else ""
            )
        )
    result["materialization"] = {
        "status": "built_and_independently_inspected",
        "plan_path": plan.as_posix(),
        "plan_sha256_before": plan_hash_before,
        "plan_sha256_after": plan_hash_after,
        "trusted_blend_path": destination.as_posix(),
        "trusted_blend_sha256_after_build": blend_hash,
        "build_report_path": build_report_path.as_posix(),
        "build_report": build_report,
    }
    _write_json(inspection, result)
    return result


def inspect_sanitized_blend(
    *,
    blend_path: Path,
    expected_registry_path: Path | None,
    out_path: Path,
    blender_bin: Path,
    timeout_seconds: int = 900,
) -> dict[str, Any]:
    """Inspect a benchmark-owned blend without saving or rendering it."""

    return _inspect_blend(
        blend_path=blend_path,
        expected_registry_path=expected_registry_path,
        catalog_plan_path=None,
        out_path=out_path,
        blender_bin=blender_bin,
        timeout_seconds=timeout_seconds,
        mode="sanitized",
    )


def inspect_registered_native_blend(
    *,
    blend_path: Path,
    registry_path: Path,
    catalog_plan_path: Path,
    out_path: Path,
    blender_bin: Path,
    timeout_seconds: int = 900,
) -> dict[str, Any]:
    """Inspect a registered native placement scene as read-only input.

    The registry supplies authoritative instance-to-root identity while the
    catalog plan supplies the expected frozen asset and transform semantics.
    Appearance state from this source is only inventoried; it is never copied
    to the sanitized evaluator scene.
    """

    return _inspect_blend(
        blend_path=blend_path,
        expected_registry_path=registry_path,
        catalog_plan_path=catalog_plan_path,
        out_path=out_path,
        blender_bin=blender_bin,
        timeout_seconds=timeout_seconds,
        mode="registered_native",
    )


def inspect_public_native_blend(
    *,
    blend_path: Path,
    instance_mapping_path: Path,
    catalog_plan_path: Path,
    out_path: Path,
    blender_bin: Path,
    timeout_seconds: int = 900,
) -> dict[str, Any]:
    """Inspect a submitter-authored fixed-catalog scene read-only.

    The unsigned mapping supplies only public instance/root identity and
    generator-owned rigid transforms. The inspector derives fingerprints; it
    never saves or renders the submitted scene.
    """

    return _inspect_blend(
        blend_path=blend_path,
        expected_registry_path=instance_mapping_path,
        catalog_plan_path=catalog_plan_path,
        out_path=out_path,
        blender_bin=blender_bin,
        timeout_seconds=timeout_seconds,
        mode="public_native",
    )


def _inspect_blend(
    *,
    blend_path: Path,
    expected_registry_path: Path | None,
    catalog_plan_path: Path | None,
    out_path: Path,
    blender_bin: Path,
    timeout_seconds: int,
    mode: str,
) -> dict[str, Any]:
    source = _required_file(blend_path, "Blender scene")
    expected = (
        _required_file(expected_registry_path, "instance registry")
        if expected_registry_path is not None
        else None
    )
    catalog_plan = (
        _required_file(catalog_plan_path, "catalog materialization plan")
        if catalog_plan_path is not None
        else None
    )
    destination = Path(out_path).expanduser().resolve()
    executable = _blender_executable(blender_bin)
    compared_paths = [source, destination]
    if expected is not None:
        compared_paths.append(expected)
    if catalog_plan is not None:
        compared_paths.append(catalog_plan)
    _require_distinct_paths(*compared_paths)
    destination.parent.mkdir(parents=True, exist_ok=True)

    source_hash_before = _sha256_file(source)
    expected_hash_before = _sha256_file(expected) if expected is not None else None
    catalog_hash_before = (
        _sha256_file(catalog_plan) if catalog_plan is not None else None
    )
    worker = Path(__file__).with_name("blend_inspector_worker.py").resolve()
    command = [
        str(executable),
        "--background",
        "--factory-startup",
        "--disable-autoexec",
        str(source),
        "--python-exit-code",
        "1",
        "--python",
        str(worker),
        "--",
        "--out-json",
        str(destination),
        "--mode",
        mode,
    ]
    if expected is not None:
        command.extend(["--expected-registry-json", str(expected)])
    if catalog_plan is not None:
        command.extend(["--catalog-plan-json", str(catalog_plan)])

    completed = None
    worker_error: Exception | None = None
    try:
        completed = _run(
            command,
            timeout_seconds=timeout_seconds,
            stdout_path=destination.with_suffix(destination.suffix + ".stdout.log"),
            stderr_path=destination.with_suffix(destination.suffix + ".stderr.log"),
            label="read-only blend inspector",
        )
    except Exception as exc:
        worker_error = exc
    finally:
        # Always re-read all source bytes, including after timeout or failure.
        source_hash_after = _sha256_file(source)
        expected_hash_after = _sha256_file(expected) if expected is not None else None
        catalog_hash_after = (
            _sha256_file(catalog_plan) if catalog_plan is not None else None
        )
    if source_hash_after != source_hash_before:
        raise MaterializationError("read-only blend inspector modified the source blend")
    if expected_hash_after != expected_hash_before:
        raise MaterializationError("read-only blend inspector modified the instance registry")
    if catalog_hash_after != catalog_hash_before:
        raise MaterializationError(
            "read-only blend inspector modified the catalog materialization plan"
        )
    if worker_error is not None:
        raise worker_error
    assert completed is not None
    if completed.returncode != 0:
        _raise_worker_error("read-only blend inspector", completed)

    result = _load_json_object(destination, "blend inspection report")
    result["source_integrity"] = {
        "source_blend_path": source.as_posix(),
        "source_blend_sha256_before": source_hash_before,
        "source_blend_sha256_after": source_hash_after,
        "source_blend_modified": False,
        "expected_registry_sha256_before": expected_hash_before,
        "expected_registry_sha256_after": expected_hash_after,
        "catalog_plan_sha256_before": catalog_hash_before,
        "catalog_plan_sha256_after": catalog_hash_after,
        "auto_execution_disabled": True,
        "source_scene_saved": False,
    }
    # Stable aliases retained for the native-placement preparation boundary.
    result["source_sha256_before"] = source_hash_before
    result["source_sha256_after"] = source_hash_after
    result["source_modified"] = False
    _write_json(destination, result)
    return result


def _run(
    command: list[str],
    *,
    timeout_seconds: int,
    stdout_path: Path,
    stderr_path: Path,
    label: str,
) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=max(1, int(timeout_seconds)),
        )
    except subprocess.TimeoutExpired as exc:
        stdout_path.write_text(_stream_text(exc.stdout), encoding="utf-8")
        stderr_path.write_text(_stream_text(exc.stderr), encoding="utf-8")
        raise MaterializationError(
            f"{label} timed out after {max(1, int(timeout_seconds))} seconds"
        ) from exc
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    return completed


def _raise_worker_error(
    label: str,
    completed: subprocess.CompletedProcess[str],
) -> None:
    detail = (completed.stderr or completed.stdout)[-4000:]
    raise MaterializationError(
        f"{label} exited with code {completed.returncode}: {detail}"
    )


def _blender_executable(path: Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise MaterializationError(f"Blender executable does not exist: {resolved}")
    if not os.access(resolved, os.X_OK):
        raise MaterializationError(f"Blender path is not executable: {resolved}")
    return resolved


def _required_file(path: Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise MaterializationError(f"{label} does not exist: {resolved}")
    return resolved


def _require_distinct_paths(*paths: Path) -> None:
    normalized = [Path(path).expanduser().resolve() for path in paths]
    if len(normalized) != len(set(normalized)):
        raise MaterializationError(
            "materialization input and output paths must be distinct"
        )


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise MaterializationError(f"{label} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MaterializationError(f"invalid {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MaterializationError(f"{label} must contain a JSON object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise MaterializationError(
            f"cannot serialize materialization report {path}: {exc}"
        ) from exc
    path.write_text(encoded, encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stream_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)
